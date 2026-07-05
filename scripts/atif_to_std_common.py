"""Shared ATIF-to-standardized-ATIF normalization helpers."""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from schema.atif import (
    ATIFTrajectory,
    Step,
    ToolCall,
    content_to_text,
    normalize_atif_trajectory,
)

SYSTEM_PROMPT_PREFIXES = (
    "You are web shopping.\n",
    "You are an agent for science world.",
    "Interact with a household to solve a task.",
)

RESPONSE_FORMAT_PROMPT = re.compile(
    r"\nYour response should use the following format:.*\Z",
    re.DOTALL,
)
TERMINAL_TASK_DESCRIPTION_MARKER = "\n\nTask Description:\n"


def _string_arg(arguments: dict[str, Any], *names: str) -> str:
    for name in names:
        value = arguments.get(name)
        if value is not None:
            return str(value)
    return ""


def _normalize_edit_tool_input(tool_input: Any) -> dict[str, Any] | None:
    if not isinstance(tool_input, dict):
        return None
    path = _string_arg(tool_input, "path", "file_path")
    if "old_string" in tool_input or "new_string" in tool_input:
        return {
            "command": "str_replace",
            "path": path,
            "old_str": _string_arg(tool_input, "old_string", "old_str"),
            "new_str": _string_arg(tool_input, "new_string", "new_str"),
        }
    if "content" in tool_input or "file_text" in tool_input:
        return {
            "command": "create",
            "path": path,
            "file_text": _string_arg(tool_input, "content", "file_text"),
        }
    return None


def standardize_tool_call(tool_call: ToolCall) -> ToolCall:
    normalized = tool_call.model_copy(deep=True)
    function_name = normalized.function_name
    arguments = dict(normalized.arguments)
    if function_name in {"execute_bash", "bash"}:
        normalized.function_name = "terminal"
        normalized.arguments = {"command": _string_arg(arguments, "command", "code")}
    elif function_name == "execute_ipython_cell":
        normalized.function_name = "python"
        normalized.arguments = {"code": _string_arg(arguments, "code", "command")}
    elif function_name in {"str_replace_editor", "file_editor"}:
        normalized.function_name = "file_editor"
        normalized.arguments = arguments
    elif function_name == "edit_file":
        normalized.function_name = "file_editor"
        normalized.arguments = {
            "command": "str_replace",
            "path": _string_arg(arguments, "path"),
            "old_str": _string_arg(arguments, "old_str"),
            "new_str": _string_arg(arguments, "content", "new_str"),
        }
    elif function_name == "generic_tool":
        raw_tool = str(arguments.get("tool_name") or "").lower()
        tool_input = arguments.get("tool_input")
        if raw_tool in {"edit", "write", "read"}:
            edit_arguments = _normalize_edit_tool_input(tool_input)
            if edit_arguments is not None:
                normalized.function_name = "file_editor"
                normalized.arguments = edit_arguments
        elif raw_tool == "bash" and isinstance(tool_input, dict):
            normalized.function_name = "terminal"
            normalized.arguments = {"command": _string_arg(tool_input, "command")}
        elif raw_tool == "websearch" and isinstance(tool_input, dict):
            normalized.function_name = "web_search"
            normalized.arguments = {"query": _string_arg(tool_input, "query")}
    elif function_name in {"submit", "stop"}:
        normalized.function_name = "finish"
        normalized.arguments = {
            "message": _string_arg(arguments, "message", "output"),
            "task_completed": True,
        }
    return normalized


def normalize_prompt_boilerplate(trajectory: ATIFTrajectory) -> None:
    for step in trajectory.steps:
        if step.source != "user" or not isinstance(step.message, str):
            continue
        if step.message.startswith(SYSTEM_PROMPT_PREFIXES):
            step.source = "system"
            continue
        step.message = RESPONSE_FORMAT_PROMPT.sub("", step.message).strip()


def renumber_steps(steps: list[Step]) -> list[Step]:
    for index, step in enumerate(steps, start=1):
        step.step_id = index
    return steps


def split_terminal_task_description_prompt(trajectory: ATIFTrajectory) -> bool:
    if not trajectory.steps:
        return False
    first_step = trajectory.steps[0]
    if first_step.source != "user" or not isinstance(first_step.message, str):
        return False
    if TERMINAL_TASK_DESCRIPTION_MARKER not in first_step.message:
        return False
    system_prompt, task_prompt = first_step.message.split(TERMINAL_TASK_DESCRIPTION_MARKER, 1)
    first_step.source = "system"
    first_step.message = system_prompt.strip()
    trajectory.steps.insert(
        1,
        Step(
            step_id=0,
            source="user",
            message=f"Task Description:\n{task_prompt.strip()}",
        ),
    )
    trajectory.steps = renumber_steps(trajectory.steps)
    return True


def structure_terminal_completion_step(step: Step) -> bool:
    if step.source != "agent" or step.tool_calls or step.observation is not None:
        return False
    text = content_to_text(step.message).strip()
    if not text.startswith("{"):
        return False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict) or payload.get("task_complete") is not True:
        return False
    commands = payload.get("commands") or []
    if not isinstance(commands, list) or any(
        isinstance(command, dict) and str(command.get("keystrokes") or "").strip()
        for command in commands
    ):
        return False
    message_parts = [
        str(payload.get("analysis") or "").strip(),
        str(payload.get("plan") or "").strip(),
    ]
    message = "\n\n".join(part for part in message_parts if part)
    if not message:
        return False
    step.message = message
    step.tool_calls = [
        ToolCall(
            tool_call_id="call_1",
            function_name="finish",
            arguments={"message": message, "task_completed": True},
            extra={"raw_format": "terminal_json"},
        )
    ]
    return True


def is_empty_message_only_step(step: Step) -> bool:
    return (
        step.observation is None
        and not step.tool_calls
        and not content_to_text(step.message).strip()
    )


def standardize_tools(trajectory: ATIFTrajectory) -> ATIFTrajectory:
    normalized = trajectory.model_copy(deep=True)
    normalize_prompt_boilerplate(normalized)
    for step in normalized.steps:
        if step.tool_calls:
            step.tool_calls = [standardize_tool_call(tool_call) for tool_call in step.tool_calls]
    return normalized


def normalize_terminal_trajectory(trajectory: ATIFTrajectory) -> ATIFTrajectory:
    normalized = standardize_tools(normalize_atif_trajectory(trajectory))
    split_terminal_task_description_prompt(normalized)
    for step in normalized.steps:
        structure_terminal_completion_step(step)
    normalized.steps = renumber_steps(
        [step for step in normalized.steps if not is_empty_message_only_step(step)]
    )
    return normalized


def main(script_file: str | None = None) -> None:  # noqa: ARG001
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        trajectory = ATIFTrajectory(**json.loads(line))
        normalized = standardize_tools(normalize_atif_trajectory(trajectory))
        print(normalized.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
