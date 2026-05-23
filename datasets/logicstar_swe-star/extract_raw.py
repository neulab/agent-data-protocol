import json
import os

from datasets import load_dataset


def emit(item):
    try:
        print(json.dumps(item, ensure_ascii=False))
    except BrokenPipeError:
        os._exit(0)


def main():
    dataset = load_dataset("LogicStar/SWE-Star", split="train", streaming=True)
    seen_ids = {}
    for item in dataset:
        if not item["resolved"]:
            continue
        instance_id = str(item["instance_id"])
        duplicate_count = seen_ids.get(instance_id, 0)
        seen_ids[instance_id] = duplicate_count + 1
        if duplicate_count:
            item["instance_id"] = f"{instance_id}_{duplicate_count}"
        emit(item)


if __name__ == "__main__":
    main()
