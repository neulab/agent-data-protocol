from __future__ import annotations

import json
import re
import sys

from scripts.raw_to_atif_common import trajectory_from_record

TARGET_TASK_RE = re.compile(
    r"^## Target Task\s*\n(?P<task>.*?)(?=^## |\Z)",
    re.DOTALL | re.MULTILINE,
)


def extract_task_instruction(content: str) -> str:
    match = TARGET_TASK_RE.search(content)
    if match:
        return match.group("task").strip()
    return content.strip()


def atif_record(record: dict) -> dict:
    messages = []
    for message in record.get("messages", []):
        role = message.get("role")
        if role == "system":
            continue
        if role == "user":
            message = {**message, "content": extract_task_instruction(message.get("content", ""))}
        messages.append(message)
    return {**record, "messages": messages}


def main(script_file: str) -> None:
    dataset_name = script_file.rsplit("/", 2)[-2]
    for index, line in enumerate(sys.stdin):
        if not line.strip():
            continue
        record = json.loads(line)
        trajectory = trajectory_from_record(atif_record(record), index, dataset_name)
        print(trajectory.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
