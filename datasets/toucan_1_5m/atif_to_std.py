# ruff: noqa: E402, I001

import json
import re
import sys
from typing import Any

from schema.atif import ATIFTrajectory, Step, normalize_atif_trajectory
from scripts.atif_to_std_common import standardize_tools
from scripts.raw_to_atif_common import renumber_steps, text_from_content

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


def remove_template_tool_declaration(trajectory: ATIFTrajectory) -> None:
    if not trajectory.steps:
        return
    tools = tool_declaration(trajectory.steps[0])
    if tools is None:
        return
    existing_tools = trajectory.agent.tool_definitions or []
    by_name = {
        tool.get("function", {}).get("name"): tool
        for tool in existing_tools
        if isinstance(tool, dict) and tool.get("function", {}).get("name")
    }
    for tool in tools:
        name = tool.get("function", {}).get("name")
        if name:
            by_name[name] = tool
    trajectory.agent.tool_definitions = list(by_name.values())
    trajectory.steps = trajectory.steps[1:]


def normalize_toucan(trajectory: ATIFTrajectory) -> ATIFTrajectory:
    normalized = standardize_tools(normalize_atif_trajectory(trajectory))
    remove_template_tool_declaration(normalized)
    normalized.steps = renumber_steps(
        [step for step in normalized.steps if not is_empty_agent_placeholder(step)]
    )
    return normalized


def main(script_file: str | None = None) -> None:  # noqa: ARG001
    for line in sys.stdin:
        if not line.strip():
            continue
        trajectory = ATIFTrajectory(**json.loads(line))
        print(normalize_toucan(trajectory).model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
