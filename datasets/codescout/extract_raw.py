import json
import os

from datasets import load_dataset

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
