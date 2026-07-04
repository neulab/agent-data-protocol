from __future__ import annotations

import json
import sys
from typing import Any

from schema.atif import Step
from scripts.raw_to_atif_common import trajectory_from_record

TASK_DESCRIPTION_MARKER = "\n\nTask Description:\n"


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
    for index, step in enumerate(trajectory.steps, start=1):
        step.step_id = index


def main(script_file: str) -> None:
    dataset_name = script_file.rsplit("/", 2)[-2]
    for index, line in enumerate(sys.stdin):
        if not line.strip():
            continue
        record: Any = json.loads(line)
        if not isinstance(record, dict):
            continue
        trajectory = trajectory_from_record(record, index, dataset_name)
        split_initial_prompt(trajectory)
        print(trajectory.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
