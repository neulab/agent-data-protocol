# ruff: noqa: E402, I001

import json
import sys

from schema.atif import ATIFTrajectory, Step, normalize_atif_trajectory
from scripts.atif_to_std_common import standardize_tools
from scripts.raw_to_atif_common import renumber_steps, text_from_content


def is_placeholder_system_message(step: Step) -> bool:
    return (
        step.source == "system"
        and not step.tool_calls
        and step.observation is None
        and text_from_content(step.message).strip() in {"", "None"}
    )


def normalize_openresearcher(trajectory: ATIFTrajectory) -> ATIFTrajectory:
    normalized = standardize_tools(normalize_atif_trajectory(trajectory))
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
