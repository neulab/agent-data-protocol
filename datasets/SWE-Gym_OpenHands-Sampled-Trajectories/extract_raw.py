import json

from datasets import load_dataset

dataset = load_dataset("SWE-Gym/OpenHands-Sampled-Trajectories")
for item in dataset["train.raw"]:
    print(json.dumps(item))
