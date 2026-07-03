import json

from datasets import load_dataset

dataset = load_dataset("osunlp/Mind2Web", split="train")

for sample in dataset:
    print(json.dumps(sample))
