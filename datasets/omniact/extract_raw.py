import json
import os

from datasets import load_dataset

dataset = load_dataset(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "omniact.py"),
    trust_remote_code=True,
)

for sample in dataset["train"]:
    print(json.dumps(sample))
