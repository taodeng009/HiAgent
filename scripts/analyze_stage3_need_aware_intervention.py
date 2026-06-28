"""Analyze Stage 3 need-aware memory intervention diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


def load_json_stream(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    records = []
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        records.append(obj)
        idx = end
    return records


def infer_task_type(task_name: str) -> str:
    name = (task_name or "").lower()
    if "pick_clean" in name:
        return "clean"
    if "pick_cool" in name:
        return "cool"
    if "pick_heat" in name:
        return "heat"
    if "look_at" in name:
        return "look"
    if "pick_two" in name:
        return "puttwo"
    if "pick_and_place" in name:
        return "place"
    return name.split("-", 1)[0] if name else "unknown"


def _avg(values: Iterable[float]) -> float | None:
    values = list(values)
    return mean(values) if values else None


def episode_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    diagnostics = record.get("agent_diagnostics") or {}
    intervention = diagnostics.get("gmemory_intervention") or {}
    metrics = intervention.get("metrics") or {}
    injection_events = intervention.get("injection_events") or []
    clear_events = intervention.get("clear_events") or []
    action_stats = diagnostics.get("alfworld_action_stats") or {}
    deltas = [float(event.get("post_injection_progress_delta") or 0.0) for event in injection_events]
    ttl_deltas = [float(event.get("delta_within_ttl") or 0.0) for event in injection_events]
    ttl_plus_3_deltas = [float(event.get("delta_within_ttl_plus_3") or 0.0) for event in injection_events]
    eventual_deltas = [float(event.get("eventual_delta_after_injection") or 0.0) for event in injection_events]
    task_start_events = [
        event
        for event in injection_events
        if event.get("phase") in {"task_start", "task_start_persistent"}
    ]
    stuck_events = [event for event in injection_events if event.get("phase") == "stuck"]
    task_start_deltas = [
        float(event.get("post_injection_progress_delta") or 0.0)
        for event in task_start_events
    ]
    stuck_deltas = [
        float(event.get("post_injection_progress_delta") or 0.0)
        for event in stuck_events
    ]
    return {
        "id": record.get("id"),
        "task_name": record.get("task_name", ""),
        "task_type": infer_task_type(record.get("task_name", "")),
        "selected_intervention_policy": intervention.get("selected_intervention_policy"),
        "policy_resolution_source": intervention.get("policy_resolution_source"),
        "success": bool(record.get("is_done")),
        "progress_rate": float(record.get("progress_rate") or 0.0),
        "retrieved": bool(intervention.get("retrieved")),
        "cached": bool(intervention.get("cached")),
        "visible_final": bool(intervention.get("visible")),
        "retrieved_but_not_visible": bool(intervention.get("retrieved_but_not_injected")),
        "injection_count": len(injection_events),
        "task_start_injection_count": len(task_start_events),
        "stuck_reactivation_count": len(stuck_events),
        "clear_count": len(clear_events),
        "ttl_expired_count": sum(1 for event in clear_events if str(event.get("reason") or "").endswith("ttl_expired")),
        "task_start_ttl_expired_count": sum(
            1 for event in clear_events if event.get("reason") == "task_start_ttl_expired"
        ),
        "stuck_ttl_expired_count": sum(
            1
            for event in clear_events
            if event.get("reason") == "ttl_expired" and event.get("phase") == "stuck"
        ),
        "cooldown_block_count": int(metrics.get("cooldown_block_count") or 0),
        "trigger_no_cached_memory_count": int(metrics.get("trigger_no_cached_memory_count") or 0),
        "trigger_no_usable_memory_count": int(metrics.get("trigger_no_usable_memory_count") or 0),
        "stuck_trigger_count": int(metrics.get("stuck_trigger_count") or 0),
        "ineffective_reactivation_streak": int(metrics.get("ineffective_reactivation_streak") or 0),
        "effective_reactivation_count": int(metrics.get("effective_reactivation_count") or 0),
        "suppressed_reactivation_count": int(metrics.get("suppressed_reactivation_count") or 0),
        "post_injection_progress_delta": max(deltas) if deltas else None,
        "post_task_start_progress_delta": max(task_start_deltas) if task_start_deltas else None,
        "post_stuck_reactivation_progress_delta": max(stuck_deltas) if stuck_deltas else None,
        "delta_within_ttl": max(ttl_deltas) if ttl_deltas else None,
        "delta_within_ttl_plus_3": max(ttl_plus_3_deltas) if ttl_plus_3_deltas else None,
        "eventual_delta_after_injection": max(eventual_deltas) if eventual_deltas else None,
        "recovery_success": any(delta > 0 for delta in deltas),
        "task_start_progress_success": any(delta > 0 for delta in task_start_deltas),
        "stuck_recovery_success": any(delta > 0 for delta in stuck_deltas),
        "memory_harm_proxy": any(delta < 0 for delta in deltas),
        "memory_exposure_steps_total": int(metrics.get("memory_exposure_steps_total") or 0),
        "memory_exposure_rate": float(metrics.get("memory_exposure_rate") or 0.0),
        "first_stuck_reactivation_step": metrics.get("first_stuck_reactivation_step"),
        "trigger_reasons": [event.get("trigger_reason") or event.get("reason", "") for event in injection_events],
        "nothing_happens_count": int(action_stats.get("nothing_happens_count") or 0),
        "check_valid_actions_count": int(action_stats.get("check_valid_actions_count") or 0),
    }


def aggregate_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(summaries)
    injection_episodes = [row for row in summaries if row["injection_count"] > 0]
    task_start_episodes = [row for row in summaries if row["task_start_injection_count"] > 0]
    stuck_reactivation_episodes = [row for row in summaries if row["stuck_reactivation_count"] > 0]
    injected_events = sum(row["injection_count"] for row in summaries)
    resolution_source_counts: Dict[str, int] = {}
    for row in summaries:
        source = str(row.get("policy_resolution_source") or "missing")
        resolution_source_counts[source] = resolution_source_counts.get(source, 0) + 1
    baseline_rows = [row for row in summaries if "baseline_success" in row]
    return {
        "episode_count": total,
        "success_count": sum(1 for row in summaries if row["success"]),
        "success_rate": sum(1 for row in summaries if row["success"]) / total if total else 0.0,
        "avg_progress_rate": _avg(row["progress_rate"] for row in summaries),
        "selected_policy_missing_count": sum(
            1 for row in summaries if not row.get("selected_intervention_policy")
        ),
        "policy_resolution_source_counts": dict(sorted(resolution_source_counts.items())),
        "retrieved_count": sum(1 for row in summaries if row["retrieved"]),
        "cached_count": sum(1 for row in summaries if row["cached"]),
        "visible_final_count": sum(1 for row in summaries if row["visible_final"]),
        "retrieved_but_not_visible_count": sum(1 for row in summaries if row["retrieved_but_not_visible"]),
        "injected_episode_count": len(injection_episodes),
        "task_start_injected_episode_count": len(task_start_episodes),
        "stuck_reactivation_episode_count": len(stuck_reactivation_episodes),
        "memory_injection_rate": len(injection_episodes) / total if total else 0.0,
        "task_start_injection_rate": len(task_start_episodes) / total if total else 0.0,
        "stuck_reactivation_rate": len(stuck_reactivation_episodes) / total if total else 0.0,
        "injected_event_count": injected_events,
        "task_start_injected_event_count": sum(row["task_start_injection_count"] for row in summaries),
        "stuck_reactivation_event_count": sum(row["stuck_reactivation_count"] for row in summaries),
        "ttl_expired_count": sum(row["ttl_expired_count"] for row in summaries),
        "task_start_ttl_expired_count": sum(row["task_start_ttl_expired_count"] for row in summaries),
        "stuck_ttl_expired_count": sum(row["stuck_ttl_expired_count"] for row in summaries),
        "cooldown_block_count": sum(row["cooldown_block_count"] for row in summaries),
        "trigger_no_cached_memory_count": sum(row["trigger_no_cached_memory_count"] for row in summaries),
        "trigger_no_usable_memory_count": sum(row["trigger_no_usable_memory_count"] for row in summaries),
        "stuck_trigger_count": sum(row["stuck_trigger_count"] for row in summaries),
        "ineffective_reactivation_streak_total": sum(row["ineffective_reactivation_streak"] for row in summaries),
        "effective_reactivation_count": sum(row["effective_reactivation_count"] for row in summaries),
        "suppressed_reactivation_count": sum(row["suppressed_reactivation_count"] for row in summaries),
        "suppressed_reactivation_episode_count": sum(
            1 for row in summaries if row["suppressed_reactivation_count"] > 0
        ),
        "recovery_success_rate": (
            sum(1 for row in injection_episodes if row["recovery_success"]) / len(injection_episodes)
            if injection_episodes
            else None
        ),
        "task_start_progress_success_rate": (
            sum(1 for row in task_start_episodes if row["task_start_progress_success"]) / len(task_start_episodes)
            if task_start_episodes
            else None
        ),
        "stuck_recovery_success_rate": (
            sum(1 for row in stuck_reactivation_episodes if row["stuck_recovery_success"]) / len(stuck_reactivation_episodes)
            if stuck_reactivation_episodes
            else None
        ),
        "memory_harm_proxy_rate": (
            sum(1 for row in injection_episodes if row["memory_harm_proxy"]) / len(injection_episodes)
            if injection_episodes
            else None
        ),
        "avg_post_injection_progress_delta": _avg(
            row["post_injection_progress_delta"] for row in injection_episodes if row["post_injection_progress_delta"] is not None
        ),
        "avg_post_task_start_progress_delta": _avg(
            row["post_task_start_progress_delta"]
            for row in task_start_episodes
            if row["post_task_start_progress_delta"] is not None
        ),
        "avg_post_stuck_reactivation_progress_delta": _avg(
            row["post_stuck_reactivation_progress_delta"]
            for row in stuck_reactivation_episodes
            if row["post_stuck_reactivation_progress_delta"] is not None
        ),
        "avg_delta_within_ttl": _avg(
            row["delta_within_ttl"] for row in injection_episodes if row["delta_within_ttl"] is not None
        ),
        "avg_delta_within_ttl_plus_3": _avg(
            row["delta_within_ttl_plus_3"] for row in injection_episodes if row["delta_within_ttl_plus_3"] is not None
        ),
        "avg_eventual_delta_after_injection": _avg(
            row["eventual_delta_after_injection"] for row in injection_episodes if row["eventual_delta_after_injection"] is not None
        ),
        "avg_check_valid_actions_count": _avg(row["check_valid_actions_count"] for row in summaries),
        "memory_exposure_steps_total": sum(row["memory_exposure_steps_total"] for row in summaries),
        "avg_memory_exposure_rate": _avg(row["memory_exposure_rate"] for row in summaries),
        "success_up": (
            sum(1 for row in baseline_rows if row["success"] and not row["baseline_success"])
            if baseline_rows
            else None
        ),
        "success_down": (
            sum(1 for row in baseline_rows if not row["success"] and row["baseline_success"])
            if baseline_rows
            else None
        ),
        "progress_up": (
            sum(1 for row in baseline_rows if row["progress_rate"] > row["baseline_progress_rate"])
            if baseline_rows
            else None
        ),
        "progress_down": (
            sum(1 for row in baseline_rows if row["progress_rate"] < row["baseline_progress_rate"])
            if baseline_rows
            else None
        ),
    }


def aggregate_by_task_type(summaries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in summaries:
        grouped.setdefault(row["task_type"], []).append(row)
    return {task_type: aggregate_summaries(rows) for task_type, rows in sorted(grouped.items())}


def aggregate_by_selected_policy(summaries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in summaries:
        policy = str(row.get("selected_intervention_policy") or "UNSPECIFIED")
        grouped.setdefault(policy, []).append(row)
    return {policy: aggregate_summaries(rows) for policy, rows in sorted(grouped.items())}


def _records_by_identity(records: List[Dict[str, Any]], label: str) -> Dict[tuple[Any, str], Dict[str, Any]]:
    indexed: Dict[tuple[Any, str], Dict[str, Any]] = {}
    for record in records:
        key = (record.get("id"), str(record.get("task_name") or ""))
        if key in indexed:
            raise ValueError(f"Duplicate {label} episode identity: {key!r}")
        indexed[key] = record
    return indexed


def _attach_baseline(summaries: List[Dict[str, Any]], baseline_records: List[Dict[str, Any]]) -> None:
    baseline_by_key = _records_by_identity(baseline_records, "baseline")
    summary_by_key: Dict[tuple[Any, str], Dict[str, Any]] = {}
    for summary in summaries:
        key = (summary.get("id"), str(summary.get("task_name") or ""))
        if key in summary_by_key:
            raise ValueError(f"Duplicate mixed episode identity: {key!r}")
        summary_by_key[key] = summary
    mixed_keys = set(summary_by_key)
    baseline_keys = set(baseline_by_key)
    if mixed_keys != baseline_keys:
        missing = sorted(baseline_keys - mixed_keys, key=str)
        extra = sorted(mixed_keys - baseline_keys, key=str)
        raise ValueError(
            "Mixed/baseline episode identities do not align: "
            f"missing_from_mixed={missing[:5]!r}, missing_from_baseline={extra[:5]!r}"
        )
    for key, summary in summary_by_key.items():
        baseline = baseline_by_key[key]
        summary["baseline_success"] = bool(baseline.get("is_done"))
        summary["baseline_progress_rate"] = float(baseline.get("progress_rate") or 0.0)


def analyze_records(
    records: List[Dict[str, Any]],
    baseline_records: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    summaries = [episode_summary(record) for record in records]
    if baseline_records is not None:
        _attach_baseline(summaries, baseline_records)
    return {
        "overall": aggregate_summaries(summaries),
        "by_task_type": aggregate_by_task_type(summaries),
        "by_selected_intervention_policy": aggregate_by_selected_policy(summaries),
        "episodes": summaries,
    }


def format_report(analysis: Dict[str, Any]) -> str:
    lines = ["# Stage 3 Need-Aware Intervention Analysis", "", "## Overall", ""]
    for key, value in analysis["overall"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## By Task Type", ""])
    for task_type, metrics in analysis["by_task_type"].items():
        lines.append(f"### {task_type}")
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.extend(["## By Selected Intervention Policy", ""])
    for policy, metrics in analysis["by_selected_intervention_policy"].items():
        lines.append(f"### {policy}")
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="Path to logs/alfworld.jsonl")
    parser.add_argument(
        "--baseline-jsonl",
        type=Path,
        help="Aligned HiAgent baseline log used to calculate success/progress up and down",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown")
    args = parser.parse_args()
    baseline_records = load_json_stream(args.baseline_jsonl) if args.baseline_jsonl else None
    analysis = analyze_records(load_json_stream(args.jsonl), baseline_records=baseline_records)
    if args.json:
        print(json.dumps(analysis, ensure_ascii=False, indent=2))
    else:
        print(format_report(analysis))


if __name__ == "__main__":
    main()
