import json
import os
import sys

from datasets import load_dataset

DATASET_NAME = "GAIR/daVinci-Dev"
CONFIG_NAME = "env_native"
SPLIT = "train"


def main():
    token = os.getenv("HF_TOKEN") or None
    dataset = load_dataset(
        DATASET_NAME,
        CONFIG_NAME,
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
            f"Failed to stream {DATASET_NAME}/{CONFIG_NAME}. The dataset is gated; "
            "authenticate with Hugging Face and ensure access has been granted.",
            file=sys.stderr,
        )
        raise exc
