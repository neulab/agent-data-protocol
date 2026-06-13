import json
import os
import sys
from typing import Any

from datasets import load_dataset


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


# Load dataset from skill_based_easy config (most accessible)
# Other configs: skill_based_medium, skill_based_mixed
dataset = load_dataset("nvidia/Nemotron-Terminal-Corpus", "skill_based_easy", split="train")

try:
    for sample in dataset:
        print(json.dumps(json_safe(sample), ensure_ascii=False))
except BrokenPipeError:
    sys.stdout = open(os.devnull, "w")
    sys.exit(0)
