import json
from itertools import zip_longest

from datasets import load_dataset

SOURCES = [
    ("issue_localize", "hybrid-gym/issue_localize_1978i"),
    ("func_localize", "hybrid-gym/func_localize_1438i"),
    ("func_gen", "hybrid-gym/func_gen_552i"),
    ("dep_search", "hybrid-gym/dep_search_502i"),
]
SPLIT = "train"


def iter_source(task_type: str, dataset_name: str):
    dataset = load_dataset(dataset_name, split=SPLIT, streaming=True)
    for row_index, item in enumerate(dataset):
        yield {
            "id": f"{task_type}_{row_index}",
            "source_dataset": dataset_name,
            "task_type": task_type,
            "split": SPLIT,
            "row_index": row_index,
            "messages": item["messages"],
        }


def main():
    iterators = [iter_source(task_type, dataset_name) for task_type, dataset_name in SOURCES]
    try:
        for rows in zip_longest(*iterators):
            for row in rows:
                if row is not None:
                    print(json.dumps(row, ensure_ascii=False))
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    main()
