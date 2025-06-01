import json

from datasets import load_dataset

dataset = load_dataset("nebius/swe-agent-trajectories")
for item in dataset["train"]:
    print(json.dumps(item))
