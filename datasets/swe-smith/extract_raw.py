import json

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
dataset = load_dataset("SWE-bench/SWE-smith-trajectories")
ids = {}

for split in ["tool", "xml", "ticks"]:
    for item in dataset[split]:
        if not item["resolved"]:
            continue
        id = str(item["traj_id"])
        if id not in ids:
            ids[id] = 0
        item["id"] = f"{id}_{ids[id]}"
        ids[id] += 1
        print(json.dumps(item))
