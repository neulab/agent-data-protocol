import json

from datasets import load_dataset

# Load all splits from the trajectories config
dataset = load_dataset("togethercomputer/CoderForge-Preview", "trajectories")
ids = {}
split = "filtered_reward1"
for item in dataset[split]:
    id = str(item["trajectory_id"])
    if id not in ids:
        ids[id] = 0
    item["id"] = f"{id}_{ids[id]}"
    item["messages"] = json.loads(item["messages"])
    ids[id] += 1
    print(json.dumps(item))
