import json
import os
import sys
from typing import Any

from datasets import get_dataset_config_names, load_dataset


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


# Load the Toucan-1.5M dataset from Hugging Face
# ds = load_dataset("Agent-Ark/Toucan-1.5M", "SFT", split="train")
repo = "Agent-Ark/Toucan-1.5M"
try:
    row_index = 0
    for config in get_dataset_config_names(repo):
        dataset = load_dataset(repo, config, split="train", streaming=True)
        for sample in dataset:
            raw_sample = dict(sample)
            raw_sample.setdefault("id", f"toucan_sample_{row_index}")
            raw_sample.setdefault("source_config", config)
            raw_sample.setdefault("row_index", row_index)
            print(json.dumps(json_safe(raw_sample), ensure_ascii=False))
            row_index += 1
except BrokenPipeError:
    sys.stdout = open(os.devnull, "w")
    sys.exit(0)
