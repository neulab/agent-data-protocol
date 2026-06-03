"""Shared ATIF-to-standardized-ATIF normalization helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schema.atif import ATIFTrajectory, normalize_atif_trajectory


def main(script_file: str | None = None) -> None:  # noqa: ARG001
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        trajectory = ATIFTrajectory(**json.loads(line))
        normalized = normalize_atif_trajectory(trajectory)
        print(normalized.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
