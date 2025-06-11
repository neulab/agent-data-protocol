#!/usr/bin/env python3
import json
from collections import Counter

from datasets import load_dataset


def main():
    ds = load_dataset("smurty/NNetNav-6k")
    unique_ids = Counter()
    # Print each item as a separate line in jsonl format
    for item in ds["train"]:
        unique_ids[item["id"]] += 1
        item["id"] = item["id"] + f"_{unique_ids[item["id"]]-1}"
        print(json.dumps(item))


if __name__ == "__main__":
    main()
