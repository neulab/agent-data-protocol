import json
import os
import sys
import urllib.parse
import urllib.request
from collections.abc import Iterable
from typing import Any

DATASET_NAME = "jupyter-agent/jupyter-agent-dataset"
DEFAULT_SPLIT = "non_thinking"
VIEWER_PAGE_SIZE = 100


def _load_with_huggingface_datasets(split: str) -> Iterable[dict[str, Any]] | None:
    try:
        from datasets import load_dataset  # type: ignore
    except (ImportError, AttributeError):
        return None

    try:
        return load_dataset(DATASET_NAME, split=split, streaming=True)
    except ValueError:
        # The dataset card has used both non_thinking and non-thinking spellings.
        if split == "non_thinking":
            return load_dataset(DATASET_NAME, split="non-thinking", streaming=True)
        raise


def _fetch_viewer_page(split: str, offset: int, length: int) -> dict[str, Any]:
    params = urllib.parse.urlencode(
        {
            "dataset": DATASET_NAME,
            "config": "default",
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{params}"
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def _load_with_viewer_api(split: str, max_rows: int | None) -> Iterable[dict[str, Any]]:
    offset = 0
    while True:
        length = VIEWER_PAGE_SIZE if max_rows is None else min(VIEWER_PAGE_SIZE, max_rows - offset)
        if length <= 0:
            return
        page = _fetch_viewer_page(split, offset, length)
        rows = page.get("rows", [])
        if not rows:
            return
        for item in rows:
            yield item["row"]
        offset += len(rows)
        total = page.get("num_rows_total")
        if total is not None and offset >= total:
            return


def iter_rows() -> Iterable[dict[str, Any]]:
    split = os.getenv("JUPYTER_AGENT_SPLIT", DEFAULT_SPLIT)
    max_rows_env = os.getenv("MAX_ROWS")
    max_rows = int(max_rows_env) if max_rows_env else None
    max_notebook_chars_env = os.getenv("MAX_ORIGINAL_NOTEBOOK_CHARS")
    max_notebook_chars = int(max_notebook_chars_env) if max_notebook_chars_env else None

    dataset = _load_with_huggingface_datasets(split)
    if dataset is None:
        print(
            "Falling back to the Hugging Face dataset viewer API. Install the "
            "`datasets` package for full streaming extraction.",
            file=sys.stderr,
        )
        dataset = _load_with_viewer_api(split, None)

    emitted = 0
    for row in dataset:
        row = dict(row)
        if (
            max_notebook_chars is not None
            and len(row.get("original_notebook") or "") > max_notebook_chars
        ):
            continue
        row["split"] = split
        print(json.dumps(row, ensure_ascii=False))
        emitted += 1
        if max_rows is not None and emitted >= max_rows:
            break


if __name__ == "__main__":
    iter_rows()
