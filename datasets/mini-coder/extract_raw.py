import json

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
dataset = load_dataset("ricdomolm/mini-coder-trajs-400k")

for item in dataset["train"]:
    # Only keep verified trajectories
    if not item["verified"]:
        continue

    # Create a unique ID for each trajectory
    item["id"] = item["instance_id"]
    print(json.dumps(item))
