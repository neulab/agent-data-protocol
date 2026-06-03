import json
import os
import sys

from datasets import load_dataset

DATASET_NAME = "GAIR/daVinci-Dev"
DATA_FILE = "hf://datasets/GAIR/daVinci-Dev/env-native.jsonl"
SPLIT = "train"


def main():
    token = os.getenv("HF_TOKEN") or None
    dataset = load_dataset(
        "json",
        data_files=DATA_FILE,
        split=SPLIT,
        streaming=True,
        token=token,
    )
    for item in dataset:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"Failed to stream {DATA_FILE}. The dataset is gated; "
            "authenticate with Hugging Face and ensure access has been granted.",
            file=sys.stderr,
        )
        raise exc
