# ruff: noqa: E402, I001

import json
import sys

from schema.atif import ATIFTrajectory, normalize_atif_trajectory
from scripts.atif_to_std_common import (
    is_empty_message_only_step,
    renumber_steps,
    split_terminal_task_description_prompt,
    standardize_tools,
    structure_terminal_completion_step,
)


def normalize_litecoder(trajectory: ATIFTrajectory) -> ATIFTrajectory:
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
        if not line.strip():
            continue
        trajectory = ATIFTrajectory(**json.loads(line))
        print(normalize_litecoder(trajectory).model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
