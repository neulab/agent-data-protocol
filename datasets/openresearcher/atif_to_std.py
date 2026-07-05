# ruff: noqa: E402, I001

import json
import sys

from schema.atif import ATIFTrajectory, Step, normalize_atif_trajectory
from scripts.atif_to_std_common import standardize_tools
from scripts.raw_to_atif_common import renumber_steps, text_from_content


def _copy_present(tool_input: dict, *names: str) -> dict:
    return {name: tool_input[name] for name in names if name in tool_input}


def is_placeholder_system_message(step: Step) -> bool:
    return (
        step.source == "system"
        and not step.tool_calls
        and step.observation is None
        and text_from_content(step.message).strip() in {"", "None"}
    )


def standardize_openresearcher_tool_calls(trajectory: ATIFTrajectory) -> None:
    for step in trajectory.steps:
        for tool_call in step.tool_calls or []:
            if tool_call.function_name == "web_search":
                tool_call.function_name = "search"
                continue
            if tool_call.function_name != "generic_tool":
                continue
            raw_tool_name = str(tool_call.arguments.get("tool_name") or "").lower()
            tool_input = tool_call.arguments.get("tool_input")
            if not isinstance(tool_input, dict):
                continue
            if raw_tool_name == "websearch":
                tool_call.function_name = "search"
                tool_call.arguments = _copy_present(tool_input, "query", "topn", "source")
            elif raw_tool_name == "browser.open":
                tool_call.function_name = "open"
                tool_call.arguments = _copy_present(
                    tool_input, "cursor", "id", "loc", "num_lines", "source", "view_source"
                )
            elif raw_tool_name == "browser.find":
                tool_call.function_name = "find"
                tool_call.arguments = _copy_present(tool_input, "pattern", "cursor")


def normalize_openresearcher(trajectory: ATIFTrajectory) -> ATIFTrajectory:
    normalized = normalize_atif_trajectory(trajectory)
    standardize_openresearcher_tool_calls(normalized)
    normalized = standardize_tools(normalized)
    normalized.steps = renumber_steps(
        [step for step in normalized.steps if not is_placeholder_system_message(step)]
    )
    return normalized


def main(script_file: str | None = None) -> None:  # noqa: ARG001
    for line in sys.stdin:
        if not line.strip():
            continue
        trajectory = ATIFTrajectory(**json.loads(line))
        print(normalize_openresearcher(trajectory).model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
