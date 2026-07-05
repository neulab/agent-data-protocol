# ruff: noqa: E402, I001

import json
import re
import sys
from copy import deepcopy
from typing import Any

from schema.atif import Step
from scripts.raw_to_atif_common import (
    dataset_name_from_script,
    parse_arguments,
    renumber_steps,
    text_from_content,
    trajectories_from_input,
)

TOOL_DECLARATION_PATTERN = re.compile(
    r"^<\|im_system\|>tool_declare<\|im_middle\|>(?P<tools>.*)<\|im_end\|>$",
    flags=re.DOTALL,
)


def is_empty_agent_placeholder(step: Step) -> bool:
    return (
        step.source == "agent"
        and not step.tool_calls
        and step.observation is None
        and not text_from_content(step.message).strip()
    )


def tool_declaration(step: Step) -> list[dict[str, Any]] | None:
    if step.source != "system" or not isinstance(step.message, str):
        return None
    match = TOOL_DECLARATION_PATTERN.match(step.message.strip())
    if not match:
        return None
    tools = json.loads(match.group("tools"))
    if not isinstance(tools, list):
        return None
    return [tool for tool in tools if isinstance(tool, dict)]


def remove_template_tool_declaration(trajectory) -> None:
    if not trajectory.steps:
        return
    tools = tool_declaration(trajectory.steps[0])
    if tools is None:
        return
    if not trajectory.agent.tool_definitions:
        trajectory.agent.tool_definitions = tools
    trajectory.steps = trajectory.steps[1:]


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
        remove_template_tool_declaration(trajectory)
        trajectory.steps = renumber_steps(
            [step for step in trajectory.steps if not is_empty_agent_placeholder(step)]
        )
        print(trajectory.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
