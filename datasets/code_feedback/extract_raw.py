import json

from datasets import load_dataset

dataset = load_dataset("m-a-p/Code-Feedback", split="train")

for sample in dataset:
    print(json.dumps(sample))
