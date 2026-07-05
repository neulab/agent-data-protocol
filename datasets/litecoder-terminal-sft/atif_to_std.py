# ruff: noqa: E402, I001

import json
import sys

from schema.atif import ATIFTrajectory, Step, ToolCall, normalize_atif_trajectory
from scripts.atif_to_std_common import standardize_tools
from scripts.raw_to_atif_common import renumber_steps, text_from_content

TASK_DESCRIPTION_MARKER = "\n\nTask Description:\n"


def split_initial_prompt(trajectory: ATIFTrajectory) -> None:
    if not trajectory.steps:
        return
    first_step = trajectory.steps[0]
    if first_step.source != "user" or not isinstance(first_step.message, str):
        return
    if TASK_DESCRIPTION_MARKER not in first_step.message:
        return
    system_prompt, task_prompt = first_step.message.split(TASK_DESCRIPTION_MARKER, 1)
    first_step.source = "system"
    first_step.message = system_prompt.strip()
    trajectory.steps.insert(
        1,
        Step(
            step_id=2,
            source="user",
            message=f"Task Description:\n{task_prompt.strip()}",
        ),
    )
    trajectory.steps = renumber_steps(trajectory.steps)


def is_empty_turn(step: Step) -> bool:
    return (
        step.observation is None
        and not step.tool_calls
        and not text_from_content(step.message).strip()
    )


def structure_terminal_completion(step: Step) -> None:
    if step.source != "agent" or step.tool_calls or step.observation is not None:
        return
    text = text_from_content(step.message).strip()
    if not text.startswith("{"):
        return
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return
    if not isinstance(payload, dict) or payload.get("task_complete") is not True:
        return
    commands = payload.get("commands")
    if not isinstance(commands, list) or any(
        isinstance(command, dict) and str(command.get("keystrokes") or "").strip()
        for command in commands
    ):
        return
    message_parts = [
        str(payload.get("analysis") or "").strip(),
        str(payload.get("plan") or "").strip(),
    ]
    message = "\n\n".join(part for part in message_parts if part)
    step.message = message
    step.tool_calls = [
        ToolCall(
            tool_call_id="call_1",
            function_name="finish",
            arguments={"message": message, "task_completed": True},
            extra={"raw_format": "terminal_json"},
        )
    ]


def normalize_litecoder(trajectory: ATIFTrajectory) -> ATIFTrajectory:
    normalized = standardize_tools(normalize_atif_trajectory(trajectory))
    split_initial_prompt(normalized)
    for step in normalized.steps:
        structure_terminal_completion(step)
    normalized.steps = renumber_steps(
        [step for step in normalized.steps if not is_empty_turn(step)]
    )
    return normalized


def main(script_file: str | None = None) -> None:  # noqa: ARG001
    for line in sys.stdin:
        if not line.strip():
            continue
        trajectory = ATIFTrajectory(**json.loads(line))
        print(normalize_litecoder(trajectory).model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
