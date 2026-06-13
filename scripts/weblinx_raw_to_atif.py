from __future__ import annotations

import json
import sys
from typing import Any

from scripts.raw_to_atif_common import trajectory_from_record


def atif_record(record: dict[str, Any]) -> dict[str, Any]:
    replay = record.get("replay")
    form = record.get("form")
    if not isinstance(replay, dict):
        return record
    converted = dict(replay)
    converted["shortcode"] = record.get("shortcode")
    if isinstance(form, dict):
        converted["description"] = form.get("description")
        converted["tasks"] = form.get("tasks")
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
