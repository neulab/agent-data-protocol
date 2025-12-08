import json
import sys
from datetime import datetime

from datasets import load_dataset
from tqdm import tqdm


def default_converter(o):
    if isinstance(o, datetime):
        return o.__str__()
    else:
        return o


dataset = load_dataset("all-hands/openhands-feedback")
for i, item in enumerate(tqdm(dataset["train"], desc="Processing trajectories", file=sys.stderr)):
    for step in item.get("trajectory", []):
        if isinstance(step.get("extras"), str):
            step["extras"] = json.loads(step["extras"])
    item["id"] = str(i)
    print(json.dumps(item, default=default_converter))
