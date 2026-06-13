from __future__ import annotations

import json
import sys

from scripts.raw_to_atif_common import find_messages, message_role, trajectory_from_record


def has_valid_conversation_shape(record: dict) -> bool:
    roles = [message_role(message) for message in find_messages(record)]
    if "user" not in roles or "agent" not in roles:
        return False
    if roles and roles[0] == "agent":
        return False
    return not any(current == "agent" and following == "agent" for current, following in zip(roles, roles[1:]))


def main(script_file: str) -> None:  # noqa: ARG001
    dataset_name = script_file.rsplit("/", 2)[-2]
    for index, line in enumerate(sys.stdin):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or not has_valid_conversation_shape(record):
            continue
        trajectory = trajectory_from_record(record, index, dataset_name)
        print(trajectory.model_dump_json(exclude_none=True))
