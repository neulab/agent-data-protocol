from __future__ import annotations

import json
import sys
from typing import Any

from schema.atif import ATIF_SCHEMA_VERSION, Agent, ATIFTrajectory, Step
from scripts.raw_to_atif_common import compact_extra, record_id, text_from_content

ROLE_MAP = {
    "assistant": "agent",
    "agent": "agent",
    "system": "system",
    "developer": "system",
    "human": "user",
    "user": "user",
}


def trajectory_from_finch_record(record: dict[str, Any], index: int) -> ATIFTrajectory:
    steps: list[Step] = []
    for message in record.get("trajectory", []):
        if not isinstance(message, dict):
            continue
        role = ROLE_MAP.get(str(message.get("role", "")).lower())
        if role is None:
            continue
        steps.append(
            Step(
                step_id=len(steps) + 1,
                source=role,
                message=text_from_content(message.get("content")),
            )
        )
    if not steps:
        steps.append(
            Step(
                step_id=1,
                source="user",
                message=text_from_content(record.get("user_prompt") or record),
            )
        )
    trajectory_id = record_id(record, index, "finch_collection")
    return ATIFTrajectory(
        schema_version=ATIF_SCHEMA_VERSION,
        session_id=trajectory_id,
        trajectory_id=trajectory_id,
        agent=Agent(name="finch_collection", version="raw"),
        steps=steps,
        extra={"raw": compact_extra(record), "source_dataset": "finch_collection"},
    )


def main() -> None:
    for index, line in enumerate(sys.stdin):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            continue
        trajectory = trajectory_from_finch_record(record, index)
        print(trajectory.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main()
