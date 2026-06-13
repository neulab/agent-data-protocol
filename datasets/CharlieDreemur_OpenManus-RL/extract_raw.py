import json
import os
import sys
import urllib.parse
import urllib.request
from typing import Iterable

DATASET_NAME = "CharlieDreemur/OpenManus-RL"
CONFIG = "default"
SPLIT = "train"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100

# Emit representative rows first so `head -5` covers the major formats in the
# heterogeneous dataset: text-world actions, ScienceWorld actions, Action Input
# tool calls, direct JSON tool calls, and ToolBench-style API traces.
REPRESENTATIVE_OFFSETS = [0, 5000, 10000, 20000, 48920]


def fetch_rows(offset: int, length: int) -> dict:
    params = urllib.parse.urlencode(
        {
            "dataset": DATASET_NAME,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    with urllib.request.urlopen(f"{ROWS_URL}?{params}", timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def iter_representative_rows() -> Iterable[dict]:
    for offset in REPRESENTATIVE_OFFSETS:
        payload = fetch_rows(offset, 1)
        rows = payload.get("rows", [])
        if rows:
            yield rows[0]["row"]


def iter_all_rows(skip_ids: set[str]) -> Iterable[dict]:
    offset = 0
    total = None
    while total is None or offset < total:
        payload = fetch_rows(offset, PAGE_SIZE)
        total = payload.get("num_rows_total", total)
        rows = payload.get("rows", [])
        if not rows:
            break
        for wrapped in rows:
            row = wrapped["row"]
            if row["id"] not in skip_ids:
                yield row
        offset += len(rows)


def emit(row: dict) -> None:
    print(json.dumps(row, ensure_ascii=False))


def main() -> None:
    emitted_ids = set()
    try:
        for row in iter_representative_rows():
            emit(row)
            emitted_ids.add(row["id"])

        if os.getenv("OPENMANUS_RL_REPRESENTATIVE_ONLY"):
            return

        for row in iter_all_rows(emitted_ids):
            emit(row)
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            sys.exit(0)


if __name__ == "__main__":
    main()
