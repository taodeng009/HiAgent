"""Helpers for converting HiAgent ALFWorld logs into ReMe trajectories."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


INTERACTION_TURN_RE = re.compile(r"^Interaction Turn (\d+)$")


def parse_pretty_json_objects(path: Path | str) -> List[Dict[str, Any]]:
    """Parse a file containing consecutive pretty-printed JSON objects."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    position = 0
    records: List[Dict[str, Any]] = []
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        value, position = decoder.raw_decode(text, position)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object in {source} at offset {position}")
        records.append(value)
    return records


def extract_task_type(task_name: Any) -> str:
    text = str(task_name or "")
    return text.split("-", 1)[0] if text else ""


def raw_goal_query(goal: Any) -> str:
    return str(goal or "").strip()


def serialize_log_record_to_reme_trajectory(record: Dict[str, Any]) -> Dict[str, Any]:
    record_id = record.get("id")
    if record_id is None:
        raise ValueError("record is missing `id`")
    goal = raw_goal_query(record.get("goal"))
    if not goal:
        raise ValueError("record is missing `goal`")
    trajectory = record.get("trajectory")
    if not isinstance(trajectory, dict) or not trajectory:
        raise ValueError("record is missing `trajectory`")

    messages: List[Dict[str, str]] = []
    turns = list(iter_interaction_turns(trajectory))
    if not turns:
        raise ValueError("record has no interaction turns")

    first_turn = turns[0][1]
    initial_observation = str(first_turn.get("Observation") or "").strip()
    messages.append(
        {
            "role": "user",
            "content": f"Goal: {goal}\nInitial observation: {initial_observation}",
        }
    )

    for _, turn in turns:
        action = turn.get("Action")
        observation = turn.get("Observation")
        if action is not None:
            messages.append({"role": "assistant", "content": f"Action: {str(action).strip()}"})
        if observation is not None:
            messages.append({"role": "user", "content": f"Observation: {str(observation).strip()}"})

    trajectory_id = str(record_id)
    return {
        "request_id": trajectory_id,
        "trajectory": {
            "trajectory_id": trajectory_id,
            "messages": messages,
            "metadata": {"query": goal},
        },
        "outcome": {
            "success": bool(record.get("is_done")),
            "score": 1.0 if bool(record.get("is_done")) else 0.0,
            "progress_rate": json_scalar(record.get("progress_rate")),
        },
    }


def iter_interaction_turns(trajectory: Dict[str, Any]) -> Iterable[Tuple[int, Dict[str, Any]]]:
    parsed: List[Tuple[int, Dict[str, Any]]] = []
    for key, value in trajectory.items():
        match = INTERACTION_TURN_RE.match(str(key))
        if not match or not isinstance(value, dict):
            continue
        parsed.append((int(match.group(1)), value))
    parsed.sort(key=lambda item: item[0])
    return parsed


def is_usable_record(record: Dict[str, Any]) -> bool:
    return bool(record.get("goal")) and "is_done" in record and bool(record.get("trajectory"))


def json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)
