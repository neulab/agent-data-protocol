#!/usr/bin/env python3
import json
import os
import sys

from datasets import load_dataset
from tqdm import tqdm

SCREENSHOTS_DIR = "datasets/go-browse-wa/screenshots"


def main():
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

    print("Loading dataset from HuggingFace...", file=sys.stderr)
    ds = load_dataset("apurvaga/go-browse-wa-raw", split="train")
    print(f"Dataset loaded: {len(ds)} items", file=sys.stderr)

    # Print each item as a separate line in jsonl format
    for item in tqdm(ds, desc="Extracting screenshots", file=sys.stderr):
        # Save PIL screenshot with correct filename format
        json_data = item["json"]
        traj_num = json_data["traj_data"]["traj_num"]
        step_number = json_data["step_data"]["step_number"]
        screenshot_path = os.path.join(SCREENSHOTS_DIR, f"{traj_num:05d}-{step_number:02d}.png")
        screenshot = item["png"]
        screenshot.save(screenshot_path)
        # Print textual content as json
        print(json.dumps(json_data))


if __name__ == "__main__":
    main()
