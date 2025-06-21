import json

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
dataset = load_dataset("SWE-bench/SWE-smith-trajectories")
ids = {}

for item in dataset["train"]:
    id = str(item["instance_id"])
    if id not in ids:
        ids[id] = 0
    item["instance_id"] = f"{id}_{ids[id]}"
    ids[id] += 1
    print(json.dumps(item))
