#!/usr/bin/env python3
import json
import os

from datasets import load_dataset

SCREENSHOTS_DIR = "datasets/go-browse-wa/screenshots"


def main():
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

    ds = load_dataset("apurvaga/go-browse-wa-raw", split="train")
    # Print each item as a separate line in jsonl format
    for item in ds:
        # Save PIL screenshot
        # screenshot_path = os.path.join(SCREENSHOTS_DIR, f"{item['__key__']}.png")
        # screenshot = item["png"]
        # screenshot.save(screenshot_path)
        # Print textual content as json
        print(json.dumps(item["json"]))


if __name__ == "__main__":
    main()
