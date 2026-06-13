#!/usr/bin/env python3
"""Extract CognitiveKernel-Pro-SFT records from Hugging Face JSONL files."""

import argparse
import json
import sys
import urllib.request
from collections.abc import Iterable

DATASET_URL = "https://huggingface.co/datasets/CognitiveKernel/CognitiveKernel-Pro-SFT/resolve/main"
SOURCE_FILES = [
    "ck-pro-web.sft.jsonl",
    "docbench.sft.jsonl",
    "tablebench.sft.jsonl",
    "webwalker_subset.sft.jsonl",
]
SAMPLE_SELECTION = {
    "ck-pro-web.sft.jsonl": {0},
    "docbench.sft.jsonl": {1},
    "tablebench.sft.jsonl": {0, 5},
    "webwalker_subset.sft.jsonl": {0},
}


def iter_source(source_file: str) -> Iterable[dict]:
    url = f"{DATASET_URL}/{source_file}"
    request = urllib.request.Request(url, headers={"User-Agent": "agent-data-protocol"})
    source_name = source_file.removesuffix(".sft.jsonl")

    with urllib.request.urlopen(request) as response:
        for source_index, line in enumerate(response):
            raw = json.loads(line)
            raw["id"] = f"{source_name}_{source_index}"
            raw["source_file"] = source_file
            raw["source_index"] = source_index
            yield raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="emit five representative records used to generate sample_raw.json",
    )
    args = parser.parse_args()

    for source_file in SOURCE_FILES:
        selected = SAMPLE_SELECTION[source_file] if args.sample else None
        remaining = len(selected) if selected is not None else None

        for record in iter_source(source_file):
            if selected is not None:
                if record["source_index"] not in selected:
                    continue
                remaining -= 1

            print(json.dumps(record, ensure_ascii=False))

            if remaining == 0:
                break


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
