import json
import signal

from datasets import load_dataset

signal.signal(signal.SIGPIPE, signal.SIG_DFL)

DATASET_NAME = "Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k"


def main():
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    seen_ids = {}

    for item in dataset:
        base_id = item["instance_id"]
        count = seen_ids.get(base_id, 0)
        seen_ids[base_id] = count + 1
        item["id"] = base_id if count == 0 else f"{base_id}_{count}"
        print(json.dumps(item))


if __name__ == "__main__":
    main()
