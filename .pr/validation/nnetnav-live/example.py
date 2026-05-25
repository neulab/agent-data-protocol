from __future__ import annotations

# ruff: noqa: I001
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sdk_validation import run_dataset_validation


DATASET_NAME = 'nnetnav-live'
RECORD_INDEX = 0
RECORD_ID = 'openweb_6442'


def load_record() -> dict:
    root = Path(__file__).resolve().parents[3]
    records = json.loads(
        (root / "datasets" / DATASET_NAME / "sample_sft" / "openhands_sdk.json")
        .read_text()
    )
    record = records[RECORD_INDEX]
    if record.get("id") != RECORD_ID:
        raise RuntimeError(
            f"Expected {RECORD_ID} at index {RECORD_INDEX}, got {record.get('id')}"
        )
    return record


def main() -> None:
    run_dataset_validation(DATASET_NAME, load_record())


if __name__ == "__main__":
    main()
