from __future__ import annotations

import json
import sys
from typing import Any

from scripts.raw_to_atif_common import trajectory_from_record


def atif_record(record: dict[str, Any]) -> dict[str, Any]:
    converted = dict(record)
    answer = {
        key.split(".", 1)[1]: value for key, value in record.items() if key.startswith("Answer.")
    }
    if answer:
        converted["Answer"] = answer
        for key in list(converted):
            if key.startswith("Answer."):
                del converted[key]
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
