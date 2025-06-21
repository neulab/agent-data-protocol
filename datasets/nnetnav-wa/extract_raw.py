#!/usr/bin/env python3
import json

from datasets import load_dataset


def main():
    ds = load_dataset("stanfordnlp/nnetnav-wa", split="train", streaming=True)
    ids = {}
    # Print each item as a separate line in jsonl format
    for item in ds:
        id = str(item["id"])
        if id not in ids:
            ids[id] = 0
        item["id"] = f"{id}_{ids[id]}"
        ids[id] += 1
        print(json.dumps(item))


if __name__ == "__main__":
    main()
