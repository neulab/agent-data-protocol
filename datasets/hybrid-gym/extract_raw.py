import json
import os
import sys
from itertools import zip_longest
from typing import Any

from datasets import load_dataset

SOURCES = [
    ("issue_localize", "hybrid-gym/issue_localize_1978i"),
    ("func_localize", "hybrid-gym/func_localize_1438i"),
    ("func_gen", "hybrid-gym/func_gen_552i"),
    ("dep_search", "hybrid-gym/dep_search_502i"),
]
SPLIT = "train"


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def iter_source(task_type: str, dataset_name: str):
    dataset = load_dataset(dataset_name, split=SPLIT, streaming=True)
    for row_index, item in enumerate(dataset):
        row = dict(item)
        row.setdefault("id", f"{task_type}_{row_index}")
        row.setdefault("source_dataset", dataset_name)
        row.setdefault("task_type", task_type)
        row.setdefault("split", SPLIT)
        row.setdefault("row_index", row_index)
        yield row


def main():
    iterators = [iter_source(task_type, dataset_name) for task_type, dataset_name in SOURCES]
    try:
        for rows in zip_longest(*iterators):
            for row in rows:
                if row is not None:
                    print(json.dumps(json_safe(row), ensure_ascii=False))
    except BrokenPipeError:
        sys.stdout = open(os.devnull, "w")
        sys.exit(0)


if __name__ == "__main__":
    main()
