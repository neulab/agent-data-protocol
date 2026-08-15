# ruff: noqa: E402, I001

import copy
import json
import sys
from typing import Any

from scripts.raw_to_atif_common import (
    dataset_name_from_script,
    text_from_content,
    trajectories_from_input,
)

BROWSER_TOOL_NAMES = {
    "browser.search": "websearch",
    "browser.open": "browser.open",
    "browser.find": "browser.find",
}


def add_browser_tool_names(record: Any) -> Any:
    if not isinstance(record, dict):
        return record
    normalized = copy.deepcopy(record)
    messages = normalized.get("messages")
    if not isinstance(messages, list):
        return normalized
    for message in messages:
        if not isinstance(message, dict):
            continue
        recipient = message.get("recipient")
        if recipient not in BROWSER_TOOL_NAMES:
            continue
        if message.get("role") == "assistant" and message.get("content_type") == "code":
            message["tool_name"] = BROWSER_TOOL_NAMES[recipient]
            message["tool_input"] = text_from_content(message.get("content"))
    return normalized


def main(script_file: str) -> None:
    dataset_name = dataset_name_from_script(script_file)
    records = (add_browser_tool_names(json.loads(line)) for line in sys.stdin if line.strip())
    for trajectory in trajectories_from_input(records, dataset_name):
        print(trajectory.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
