from __future__ import annotations

import json
import sys
from typing import Any

from scripts.raw_to_atif_common import trajectory_from_record


def atif_record(record: dict[str, Any]) -> dict[str, Any]:
    converted = dict(record)
    converted["trajectory"] = []
    for step in record.get("trajectory", []):
        if not isinstance(step, dict):
            converted["trajectory"].append(step)
            continue
        converted_step = dict(step)
        if isinstance(converted_step.get("extras"), str):
            try:
                converted_step["extras"] = json.loads(converted_step["extras"])
            except json.JSONDecodeError:
                pass
        converted["trajectory"].append(converted_step)
    return converted


def main(script_file: str) -> None:
    dataset_name = script_file.rsplit("/", 2)[-2]
    for index, line in enumerate(sys.stdin):
        if not line.strip():
            continue
        raw_record = json.loads(line)
        trajectory = trajectory_from_record(atif_record(raw_record), index, dataset_name)
        print(trajectory.model_dump_json(exclude_none=True))
