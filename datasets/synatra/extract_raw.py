from datasets import load_dataset
import json


dataset = load_dataset("oottyy/Synatra",split='train')

for sample in dataset:
    print(json.dumps(sample))
