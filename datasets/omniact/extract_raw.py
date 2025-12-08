import json
import os
import sys

from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "omniact.py"),
    trust_remote_code=True,
)

for sample in tqdm(dataset["train"], desc="Processing samples", file=sys.stderr):
    print(json.dumps(sample))
