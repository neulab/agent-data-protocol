#!/usr/bin/env python3
"""
Script to indent single-line JSON files.
"""

import json
import os
from pathlib import Path


def indent_json_file(file_path):
    """Indent a JSON file if it's a single-line file."""
    with open(file_path, "r") as f:
        content = f.read().strip()

    # Check if the file is a single-line JSON file
    if content.startswith("[{") and "\n" not in content[:20]:
        print(f"Indenting {file_path}...")
        try:
            data = json.loads(content)
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in {file_path}: {e}")
            return False
    return False


def main():
    """Main function."""
    dataset_path = Path(__file__).parent / "datasets"

    # Find all sample_std.json files
    for subdir in os.listdir(dataset_path):
        subdir_path = os.path.join(dataset_path, subdir)
        sample_path = os.path.join(subdir_path, "sample_std.json")
        if os.path.exists(sample_path):
            indent_json_file(sample_path)


if __name__ == "__main__":
    main()
