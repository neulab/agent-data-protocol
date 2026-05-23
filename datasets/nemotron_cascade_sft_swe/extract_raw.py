import json
import sys
import urllib.request
from contextlib import ExitStack
from typing import Iterator, TextIO

DATASET_BASE_URL = "https://huggingface.co/datasets/nvidia/Nemotron-Cascade-SFT-SWE/resolve/main"
SOURCE_FILES = [
    "swe_localization.jsonl",
    "swe_repair.jsonl",
    "swe_testgen.jsonl",
]


def _open_source(filename: str):
    url = f"{DATASET_BASE_URL}/{filename}"
    response = urllib.request.urlopen(url, timeout=120)
    return response


def _iter_jsonl(stream, source_file: str) -> Iterator[dict]:
    stem = source_file.removesuffix(".jsonl")
    for index, line in enumerate(stream):
        if not line.strip():
            continue
        row = json.loads(line.decode("utf-8"))
        row["id"] = f"{stem}_{index}"
        yield row


def iter_rows() -> Iterator[dict]:
    """Yield rows round-robin across task files for representative head samples."""
    with ExitStack() as stack:
        streams = [stack.enter_context(_open_source(filename)) for filename in SOURCE_FILES]
        iterators = [
            iter(_iter_jsonl(stream, filename)) for stream, filename in zip(streams, SOURCE_FILES)
        ]
        active = [True] * len(iterators)

        while any(active):
            for i, iterator in enumerate(iterators):
                if not active[i]:
                    continue
                try:
                    yield next(iterator)
                except StopIteration:
                    active[i] = False


def main(output: TextIO = sys.stdout) -> None:
    for row in iter_rows():
        output.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
