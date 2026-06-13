from __future__ import annotations

import json
import sys
from typing import Any

from scripts.raw_to_atif_common import trajectory_from_record


def extract_sop(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if line and line[0].isdigit())


def atif_record(record: dict[str, Any]) -> dict[str, Any]:
    webarena = record.get("webarena")
    converted = dict(record)
    if isinstance(webarena, dict) and webarena.get("intent"):
        converted["task"] = webarena["intent"]
    if isinstance(record.get("sop_text"), str):
        converted["sop"] = extract_sop(record["sop_text"])
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
