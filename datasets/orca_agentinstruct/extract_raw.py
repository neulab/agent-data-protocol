import json
import os
import sys

from datasets import load_dataset

ds = load_dataset("microsoft/orca-agentinstruct-1M-v1")


try:
    for category in ds:
        cat_data = ds[category]
        # if category != "code_":
        #     continue
        for id, sample in enumerate(cat_data):
            sample["id"] = f"orca_{category}{id}"
            print(json.dumps(sample, ensure_ascii=False))
except BrokenPipeError:
    sys.stdout = open(os.devnull, "w")
    sys.exit(0)
