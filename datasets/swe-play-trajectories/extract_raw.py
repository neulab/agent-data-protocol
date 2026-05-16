#!/usr/bin/env python3
"""Extract raw data from the SWE-Play-trajectories dataset on HuggingFace."""

import json

from datasets import load_dataset

dataset = load_dataset("StephenZhu/SWE-Play-trajectories")

# Print each item as a separate line in jsonl format
for idx, item in enumerate(dataset["train"]):
    item["id"] = f"swe-play-{idx}"
    print(json.dumps(item))
