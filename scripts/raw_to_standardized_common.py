"""Compatibility wrapper for legacy raw_to_standardized.py entrypoints.

The standardized representation is now normalized ATIF. This helper preserves the
old command name by piping raw records through the raw-to-ATIF projection and the
ATIF normalization stage in one process.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schema.atif import normalize_atif_trajectory
from scripts.raw_to_atif_common import dataset_name_from_script, trajectories_from_input


def main(script_file: str) -> None:
    dataset_name = dataset_name_from_script(script_file)
    records = [json.loads(line) for line in sys.stdin if line.strip()]
    for trajectory in trajectories_from_input(records, dataset_name):
        normalized = normalize_atif_trajectory(trajectory)
        print(normalized.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
