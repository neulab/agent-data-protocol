# ruff: noqa: E402, I001

import json
import sys

from schema.atif import ATIFTrajectory, Step, normalize_atif_trajectory
from scripts.atif_to_std_common import (
    split_terminal_task_description_prompt,
    standardize_tools,
    structure_terminal_completion_step,
)
from scripts.raw_to_atif_common import renumber_steps, text_from_content


def is_empty_turn(step: Step) -> bool:
    return (
        step.observation is None
        and not step.tool_calls
        and not text_from_content(step.message).strip()
    )


def normalize_nemotron(trajectory: ATIFTrajectory) -> ATIFTrajectory:
    normalized = standardize_tools(normalize_atif_trajectory(trajectory))
    split_terminal_task_description_prompt(normalized)
    for step in normalized.steps:
        structure_terminal_completion_step(step)
    normalized.steps = renumber_steps(
        [step for step in normalized.steps if not is_empty_turn(step)]
    )
    return normalized


def main(script_file: str | None = None) -> None:  # noqa: ARG001
    for line in sys.stdin:
        if not line.strip():
            continue
        trajectory = ATIFTrajectory(**json.loads(line))
        print(normalize_nemotron(trajectory).model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
