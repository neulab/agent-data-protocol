"""Normalize ATIF JSONL records while keeping ATIF as both input and output.

Despite the historical script name, this does not emit ADP ``sample_std``
records. It standardizes ATIF tool names/arguments before downstream SFT
converters consume the ATIF trajectory.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schema.atif import ATIFTrajectory, normalize_atif_trajectory


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        trajectory = ATIFTrajectory(**json.loads(line))
        normalized = normalize_atif_trajectory(trajectory)
        print(normalized.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main()
