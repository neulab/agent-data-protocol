import json

from datasets import load_dataset

dataset = load_dataset("SWE-Gym/OpenHands-Sampled-Trajectories")
ids = {}

for item in dataset["train.raw"]:
    id = str(item["instance_id"])
    if id not in ids:
        ids[id] = 0
    item["instance_id"] = f"{id}_{ids[id]}"
    ids[id] += 1
    print(json.dumps(item))
