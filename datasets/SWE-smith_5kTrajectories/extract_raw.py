import sys
import json
from datetime import datetime
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
dataset = load_dataset("SWE-bench/SWE-smith-trajectories")

for item in dataset["train"]:
    print(json.dumps(item))