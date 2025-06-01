#!/usr/bin/env python3
import json

from datasets import load_dataset


def main():
    ds = load_dataset("smurty/NNetNav-6k")

    # Print each item as a separate line in jsonl format
    for item in ds["train"]:
        print(json.dumps(item))


if __name__ == "__main__":
    main()
