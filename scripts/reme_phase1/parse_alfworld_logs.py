"""Parse HiAgent ALFWorld logs and emit ReMe-normalized trajectories."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = REPO_ROOT / "agentboard" / "agents"
if str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))

from reme_trajectory import (
    extract_task_type,
    is_usable_record,
    parse_pretty_json_objects,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    run_id = args.run_id or input_path.parents[1].name
    output_dir = args.output_dir or Path("outputs/reme_phase_1") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    records = parse_pretty_json_objects(input_path)
    normalised: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    task_type_counts: Counter[str] = Counter()
    success_by_task_type: Counter[str] = Counter()

    for index, record in enumerate(records):
        task_type = extract_task_type(record.get("task_name"))
        task_type_counts[task_type] += 1
        if record.get("is_done") is True:
            success_by_task_type[task_type] += 1
        if not is_usable_record(record):
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
            continue
        try:
            normalised.append(serialize_log_record_to_reme_trajectory(record))
        except ValueError as exc:
            skipped.append(
                {
                    "index": index,
                    "record_id": record.get("id"),
                    "task_name": record.get("task_name"),
                    "error": str(exc),
                }
            )

    write_jsonl(output_dir / "normalized_trajectories.jsonl", normalised)
    summary = {
        "input": str(input_path),
        "run_id": run_id,
        "total_records": len(records),
        "normalized_records": len(normalised),
        "skipped_records": len(skipped),
        "task_type_counts": dict(sorted(task_type_counts.items())),
        "success_by_task_type": dict(sorted(success_by_task_type.items())),
        "skipped": skipped,
    }
    (output_dir / "parsed_records_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(normalised)} normalized trajectories to {output_dir}")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
