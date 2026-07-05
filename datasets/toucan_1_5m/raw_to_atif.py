# ruff: noqa: E402, I001

import json
import sys
from copy import deepcopy
from typing import Any

from scripts.raw_to_atif_common import (
    dataset_name_from_script,
    parse_arguments,
    trajectories_from_input,
)


def add_legacy_function_calls(record: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(record)
    messages = normalized.get("messages")
    if isinstance(messages, str):
        messages = json.loads(messages)
    if not isinstance(messages, list):
        return normalized
    for index, message in enumerate(messages, start=1):
        if not isinstance(message, dict) or "function_call" not in message:
            continue
        function_call = message.get("function_call")
        if not isinstance(function_call, dict) or not function_call.get("name"):
            continue
        message["tool_calls"] = [
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {
                    "name": function_call["name"],
                    "arguments": parse_arguments(function_call.get("arguments")),
                },
            }
        ]
    normalized["messages"] = messages
    return normalized


def main(script_file: str) -> None:
    dataset_name = dataset_name_from_script(script_file)
    records = (add_legacy_function_calls(json.loads(line)) for line in sys.stdin if line.strip())
    for trajectory in trajectories_from_input(records, dataset_name):
        print(trajectory.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
