import json

from datasets import load_dataset

# Load all splits from the trajectories config
dataset = load_dataset("togethercomputer/CoderForge-Preview", "trajectories")
ids = {}

for split in ["SWE_Rebench", "SWE_Smith", "R2E_Gym", "filtered_reward1"]:
    for item in dataset[split]:
        id = str(item["trajectory_id"])
        if id not in ids:
            ids[id] = 0
        item["id"] = f"{id}_{ids[id]}"
        ids[id] += 1
        print(json.dumps(item))
