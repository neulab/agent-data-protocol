from __future__ import annotations

import json
import sys

from scripts.raw_to_atif_common import find_messages, message_role, trajectory_from_record


def has_valid_conversation_shape(record: dict) -> bool:
    roles = [message_role(message) for message in find_messages(record)]
    return "user" in roles and "agent" in roles and bool(roles) and roles[0] != "agent"


def main(script_file: str) -> None:
    dataset_name = script_file.rsplit("/", 2)[-2]
    for index, line in enumerate(sys.stdin):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or not has_valid_conversation_shape(record):
            continue
        trajectory = trajectory_from_record(record, index, dataset_name)
        print(trajectory.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
