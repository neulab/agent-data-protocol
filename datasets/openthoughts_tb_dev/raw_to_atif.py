from __future__ import annotations

import json
import sys

from scripts.raw_to_atif_common import trajectory_from_record


def is_placeholder_solution(content: str) -> bool:
    normalized = content.strip().lower()
    return not normalized or "no solution written" in normalized


def has_usable_solution(record: dict) -> bool:
    solution = record.get("solution")
    if not isinstance(solution, dict):
        return False
    content = solution.get("content")
    return isinstance(content, str) and not is_placeholder_solution(content)


def main(script_file: str) -> None:
    dataset_name = script_file.rsplit("/", 2)[-2]
    for index, line in enumerate(sys.stdin):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or not has_usable_solution(record):
            continue
        trajectory = trajectory_from_record(record, index, dataset_name)
        print(trajectory.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
