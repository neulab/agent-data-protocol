# ruff: noqa: E402, I001

from __future__ import annotations

import json
import sys

from miroverse_tools import extract_available_tools_from_messages
from scripts.raw_to_atif_common import trajectory_from_record


if __name__ == "__main__":
    dataset_name = __file__.rsplit("/", 2)[-2]
    for index, line in enumerate(sys.stdin):
        if not line.strip():
            continue
        record = json.loads(line)
        trajectory = trajectory_from_record(record, index, dataset_name)
        trajectory.agent.tool_definitions = (
            extract_available_tools_from_messages(record.get("messages", [])) or None
        )
        print(trajectory.model_dump_json(exclude_none=True))
