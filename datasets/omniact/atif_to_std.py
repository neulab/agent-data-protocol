# ruff: noqa: E402, I001

import json
import sys

from schema.atif import ATIFTrajectory, normalize_atif_trajectory
from scripts.atif_to_std_common import standardize_tools
from scripts.raw_to_atif_common import renumber_steps


def normalize_omniact(trajectory: ATIFTrajectory) -> ATIFTrajectory:
    normalized = standardize_tools(normalize_atif_trajectory(trajectory))
    if (
        len(normalized.steps) >= 2
        and normalized.steps[0].source == "agent"
        and normalized.steps[0].observation is not None
        and not normalized.steps[0].tool_calls
        and normalized.steps[1].source == "user"
    ):
        normalized.steps[0], normalized.steps[1] = normalized.steps[1], normalized.steps[0]
        normalized.steps = renumber_steps(normalized.steps)
    return normalized


def main(script_file: str | None = None) -> None:  # noqa: ARG001
    for line in sys.stdin:
        if not line.strip():
            continue
        trajectory = ATIFTrajectory(**json.loads(line))
        print(normalize_omniact(trajectory).model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
