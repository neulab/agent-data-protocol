from __future__ import annotations

import json
import sys
from typing import Any

from scripts.raw_to_atif_common import (
    dataset_name_from_script,
    split_terminal_task_description_prompt,
    structure_terminal_completion_step,
    trajectory_from_record,
)


def main(script_file: str) -> None:
    dataset_name = dataset_name_from_script(script_file)
    for index, line in enumerate(sys.stdin):
        if not line.strip():
            continue
        record: Any = json.loads(line)
        if not isinstance(record, dict):
            continue
        trajectory = trajectory_from_record(record, index, dataset_name)
        split_terminal_task_description_prompt(trajectory)
        for step in trajectory.steps:
            structure_terminal_completion_step(step)
        print(trajectory.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
