from __future__ import annotations

import json
import sys

from schema.atif import Step
from scripts.raw_to_atif_common import (
    dataset_name_from_script,
    find_messages,
    message_role,
    renumber_steps,
    text_from_content,
    trajectory_from_record,
)

TASK_DESCRIPTION_MARKER = "\n\nTask Description:\n"


def has_valid_conversation_shape(record: dict) -> bool:
    roles = [message_role(message) for message in find_messages(record)]
    return "user" in roles and "agent" in roles and bool(roles) and roles[0] != "agent"


def split_initial_prompt(trajectory) -> None:
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


def main(script_file: str) -> None:
    dataset_name = dataset_name_from_script(script_file)
    for index, line in enumerate(sys.stdin):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or not has_valid_conversation_shape(record):
            continue
        trajectory = trajectory_from_record(record, index, dataset_name)
        split_initial_prompt(trajectory)
        trajectory.steps = renumber_steps(
            [step for step in trajectory.steps if not is_empty_turn(step)]
        )
        print(trajectory.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
