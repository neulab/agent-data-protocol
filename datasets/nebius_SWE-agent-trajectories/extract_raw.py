import json

from datasets import load_dataset

dataset = load_dataset("nebius/swe-agent-trajectories")
ids = {}
# Print each item as a separate line in jsonl format
for item in dataset["train"]:
    id = str(item["instance_id"])
    if id not in ids:
        ids[id] = 0
    item["instance_id"] = f"{id}_{ids[id]}"
    ids[id] += 1
    print(json.dumps(item))
