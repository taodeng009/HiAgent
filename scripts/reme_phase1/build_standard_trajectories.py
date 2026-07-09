"""Build the Phase 1 fixed-pool trajectory package for ReMe offline import."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = REPO_ROOT / "agentboard" / "agents"
if str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))

from reme_trajectory import (
    extract_task_type,
    is_usable_record,
    parse_pretty_json_objects,
    raw_goal_query,
    serialize_log_record_to_reme_trajectory,
)


DEFAULT_INPUT = Path(
    "logs/alfworld/hiagent/"
    "test134_ContextEfficientAgentV2_qwen3_5_4b_vllm_server_30/logs/alfworld.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--per-task-type", type=int, default=2)
    parser.add_argument(
        "--allow-insufficient",
        action="store_true",
        help="Use all available successes when a task type has fewer than --per-task-type samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_task_type <= 0:
        raise SystemExit("--per-task-type must be positive")

    input_path = args.input
    run_id = args.run_id or input_path.parents[1].name
    output_dir = args.output_dir or Path("outputs/reme_phase_1") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    records = parse_pretty_json_objects(input_path)
    usable_records, skipped_records = split_usable_records(records)
    selected_records, insufficient = select_pool_build_records(
        usable_records=usable_records,
        source_log=input_path,
        per_task_type=args.per_task_type,
    )
    if insufficient and not args.allow_insufficient:
        task_summary = ", ".join(
            f"{item['task_type']}={item['available']}/{item['required']}"
            for item in insufficient
        )
        raise SystemExit(
            "Insufficient successful trajectories for task types: "
            f"{task_summary}. Re-run with --allow-insufficient to use all available successes."
        )

    pool_tasks = [build_pool_task_row(record, input_path) for record in selected_records]
    standard_trajectories = [serialize_log_record_to_reme_trajectory(record) for record in selected_records]
    manifest = build_manifest(
        input_path=input_path,
        run_id=run_id,
        output_dir=output_dir,
        records=records,
        usable_records=usable_records,
        skipped_records=skipped_records,
        selected_records=selected_records,
        insufficient=insufficient,
        per_task_type=args.per_task_type,
        allow_insufficient=args.allow_insufficient,
    )

    write_json(output_dir / "manifest.json", manifest)
    write_jsonl(output_dir / "pool_build_tasks.jsonl", pool_tasks)
    write_jsonl(output_dir / "standard_trajectories.jsonl", standard_trajectories)
    print(
        f"Wrote {len(selected_records)} pool-build trajectories "
        f"across {len(set(row['task_type'] for row in pool_tasks))} task types to {output_dir}"
    )


def split_usable_records(records: List[Dict[str, Any]]):
    usable: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        if is_usable_record(record):
            usable.append(record)
            continue
        skipped.append(
            {
                "index": index,
                "record_id": record.get("id"),
                "task_name": record.get("task_name"),
                "missing_goal": not bool(record.get("goal")),
                "missing_is_done": "is_done" not in record,
                "missing_trajectory": not bool(record.get("trajectory")),
            }
        )
    return usable, skipped


def select_pool_build_records(
    usable_records: List[Dict[str, Any]],
    source_log: Path,
    per_task_type: int,
):
    successes = [record for record in usable_records if record.get("is_done") is True]
    successes.sort(key=lambda record: stable_sort_key(record, source_log))

    first_success_by_instance: Dict[str, Dict[str, Any]] = {}
    for record in successes:
        instance_id = task_instance_id(record)
        if instance_id not in first_success_by_instance:
            first_success_by_instance[instance_id] = record

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in first_success_by_instance.values():
        grouped[extract_task_type(record.get("task_name"))].append(record)
    for records in grouped.values():
        records.sort(key=lambda record: stable_sort_key(record, source_log))

    selected: List[Dict[str, Any]] = []
    insufficient: List[Dict[str, Any]] = []
    for task_type in sorted(grouped):
        records = grouped[task_type]
        if len(records) < per_task_type:
            insufficient.append(
                {
                    "task_type": task_type,
                    "available": len(records),
                    "required": per_task_type,
                }
            )
        selected.extend(records[:per_task_type])

    selected.sort(key=lambda record: stable_sort_key(record, source_log))
    return selected, insufficient


def stable_sort_key(record: Dict[str, Any], source_log: Path):
    return (
        str(source_log).replace("\\", "/"),
        str(record.get("task_name") or ""),
        numeric_or_text(record.get("id")),
        str(record.get("id") or ""),
    )


def numeric_or_text(value: Any):
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def task_instance_id(record: Dict[str, Any]) -> str:
    return str(record.get("task_name") or record.get("id"))


def build_pool_task_row(record: Dict[str, Any], source_log: Path) -> Dict[str, Any]:
    record_id = str(record.get("id"))
    task_name = str(record.get("task_name") or "")
    return {
        "trajectory_id": record_id,
        "request_id": record_id,
        "record_id": record.get("id"),
        "task_name": task_name,
        "task_type": extract_task_type(task_name),
        "task_instance_id": task_instance_id(record),
        "goal": raw_goal_query(record.get("goal")),
        "source_log": str(source_log),
        "is_done": bool(record.get("is_done")),
        "progress_rate": record.get("progress_rate"),
    }


def build_manifest(
    input_path: Path,
    run_id: str,
    output_dir: Path,
    records: List[Dict[str, Any]],
    usable_records: List[Dict[str, Any]],
    skipped_records: List[Dict[str, Any]],
    selected_records: List[Dict[str, Any]],
    insufficient: List[Dict[str, Any]],
    per_task_type: int,
    allow_insufficient: bool,
) -> Dict[str, Any]:
    task_type_counts = Counter(extract_task_type(record.get("task_name")) for record in usable_records)
    success_by_task_type = Counter(
        extract_task_type(record.get("task_name"))
        for record in usable_records
        if record.get("is_done") is True
    )
    selected_by_task_type = Counter(extract_task_type(record.get("task_name")) for record in selected_records)
    selected_refs = [build_pool_task_ref(record) for record in selected_records]
    selected_instances = sorted({task_instance_id(record) for record in selected_records})
    return {
        "schema_version": "reme_phase1_b1_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "input_log": str(input_path),
        "output_dir": str(output_dir),
        "files": {
            "manifest": "manifest.json",
            "pool_build_tasks": "pool_build_tasks.jsonl",
            "standard_trajectories": "standard_trajectories.jsonl",
        },
        "selection": {
            "per_task_type": per_task_type,
            "allow_insufficient": allow_insufficient,
            "success_only": True,
            "task_instance_id": "task_name",
            "stable_order": ["source_log", "task_name", "record_id", "trajectory_id"],
            "duplicate_instance_policy": "keep_first_success_in_stable_order",
        },
        "held_out_rule": {
            "source": "input_log",
            "exclude_task_instance_ids": selected_instances,
            "description": "Held-out tasks are usable records from input_log whose task_instance_id is not in exclude_task_instance_ids.",
        },
        "counts": {
            "total_records": len(records),
            "usable_records": len(usable_records),
            "skipped_records": len(skipped_records),
            "success_records": sum(1 for record in usable_records if record.get("is_done") is True),
            "pool_build_records": len(selected_records),
            "task_type_counts": dict(sorted(task_type_counts.items())),
            "success_by_task_type": dict(sorted(success_by_task_type.items())),
            "selected_by_task_type": dict(sorted(selected_by_task_type.items())),
        },
        "insufficient_success_task_types": insufficient,
        "pool_build_task_refs": selected_refs,
        "skipped_records": skipped_records,
        "api_boundary": {
            "hiagent_calls_api": False,
            "offline_import_expected_by_api_side": True,
        },
    }


def build_pool_task_ref(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "trajectory_id": str(record.get("id")),
        "request_id": str(record.get("id")),
        "record_id": record.get("id"),
        "task_name": record.get("task_name"),
        "task_type": extract_task_type(record.get("task_name")),
        "task_instance_id": task_instance_id(record),
    }


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
