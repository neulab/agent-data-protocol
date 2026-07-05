# ruff: noqa: E402, I001

import json
import sys

from schema.atif import ATIFTrajectory
from scripts.atif_to_std_common import normalize_terminal_trajectory


def main(script_file: str | None = None) -> None:  # noqa: ARG001
    for line in sys.stdin:
        if not line.strip():
            continue
        trajectory = ATIFTrajectory(**json.loads(line))
        print(normalize_terminal_trajectory(trajectory).model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
