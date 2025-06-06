#!/usr/bin/env python3
import glob
import json
import os


def fix_from_field(file_path):
    """Fix the 'from' field in sample_sft.json files."""
    with open(file_path, "r") as f:
        data = json.load(f)

    modified = False
    for item in data:
        if "conversations" not in item:
            continue

        for message in item["conversations"]:
            if "value" not in message:
                continue

            value = message["value"]
            is_function_call = any(
                [
                    "<function=" in value,
                    "<function_calls>" in value,
                    "<invoke name=" in value,
                ]
            )

            if is_function_call and message["from"] != "function_call":
                message["from"] = "function_call"
                modified = True

    if modified:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Fixed {file_path}")
    else:
        print(f"No changes needed for {file_path}")


def main():
    """Fix all sample_sft.json files in the agenttuning datasets."""
    datasets_dir = os.path.join(os.getcwd(), "datasets")
    sample_sft_files = glob.glob(f"{datasets_dir}/agenttuning_*/sample_sft.json")

    for file_path in sample_sft_files:
        fix_from_field(file_path)


if __name__ == "__main__":
    main()
