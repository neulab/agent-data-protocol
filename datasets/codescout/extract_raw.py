import importlib
import json
import os
import sys
from pathlib import Path


def import_hf_load_dataset():
    repo_root = Path(__file__).resolve().parents[2]
    original_path = list(sys.path)
    sys.path = [
        path
        for path in sys.path
        if Path(path or ".").resolve() != repo_root
        and Path(path or ".").resolve() != repo_root / "datasets"
    ]
    sys.modules.pop("datasets", None)
    try:
        return importlib.import_module("datasets").load_dataset
    finally:
        sys.path = original_path


load_dataset = import_hf_load_dataset()

SOURCES = [
    ("OpenHands/CodeScout_Training_Rollouts", "CodeScout_4B", "train"),
    ("OpenHands/CodeScout_Training_Rollouts", "CodeScout_14B", "train"),
    ("adityasoni17/CodeScout14B_RFT_SWE_Smith", "default", "train"),
]


def row_limit() -> int | None:
    value = os.getenv("MAX_ROWS_PER_SOURCE")
    return int(value) if value else None


def main():
    max_rows = row_limit()
    for source_dataset, source_config, source_split in SOURCES:
        dataset = load_dataset(
            source_dataset,
            source_config,
            split=source_split,
            streaming=True,
        )
        for row_id, row in enumerate(dataset):
            if max_rows is not None and row_id >= max_rows:
                break
            item = dict(row)
            item["source_dataset"] = source_dataset
            item["source_config"] = source_config
            item["source_split"] = source_split
            item["row_id"] = row_id
            print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
