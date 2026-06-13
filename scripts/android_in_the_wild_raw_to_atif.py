from __future__ import annotations

import json
import sys
from typing import Any

from scripts.raw_to_atif_common import trajectory_from_record

ACTION_TYPES = {
    3: "type",
    4: "dual-point gesture",
    5: "go_back",
    6: "go_home",
    7: "enter",
    10: "task_complete",
    11: "task_impossible",
}


def atif_record(record: dict[str, Any]) -> dict[str, Any]:
    converted = dict(record)
    action_type = converted.get("results/action_type")
    if isinstance(action_type, int) and action_type in ACTION_TYPES:
        converted["results/action_type"] = ACTION_TYPES[action_type]
    return converted


def main(script_file: str) -> None:
    dataset_name = script_file.rsplit("/", 2)[-2]
    for index, line in enumerate(sys.stdin):
        if not line.strip():
            continue
        raw_record = json.loads(line)
        trajectory = trajectory_from_record(atif_record(raw_record), index, dataset_name)
        if trajectory.extra is not None:
            trajectory.extra["raw"] = raw_record
        print(trajectory.model_dump_json(exclude_none=True))
