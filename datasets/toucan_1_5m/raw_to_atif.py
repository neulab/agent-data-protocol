# ruff: noqa: E402, I001

import json
import sys

from schema.atif import Step
from scripts.raw_to_atif_common import (
    dataset_name_from_script,
    renumber_steps,
    text_from_content,
    trajectories_from_input,
)


def is_empty_agent_placeholder(step: Step) -> bool:
    return (
        step.source == "agent"
        and not step.tool_calls
        and step.observation is None
        and not text_from_content(step.message).strip()
    )


def main(script_file: str) -> None:
    dataset_name = dataset_name_from_script(script_file)
    records = (json.loads(line) for line in sys.stdin if line.strip())
    for trajectory in trajectories_from_input(records, dataset_name):
        trajectory.steps = renumber_steps(
            [step for step in trajectory.steps if not is_empty_agent_placeholder(step)]
        )
        print(trajectory.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
