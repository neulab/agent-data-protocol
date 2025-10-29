import json
import sys
from datasets import load_dataset

dataset = load_dataset("oottyy/Synatra", split="train")

for sample in dataset:
    print(json.dumps(sample))
