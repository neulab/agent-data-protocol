#!/usr/bin/env python3
"""
Script to fix the 'from' field in sample_sft.json files.
Changes 'from': 'gpt' to 'from': 'function_call' for messages containing function calls.
"""

import json
import os


def contains_function_call(value):
    """Check if a message value contains function call patterns."""
    if not isinstance(value, str):
        return False

    function_patterns = ["<function=", "<function_calls>", "<invoke name="]

    return any(pattern in value for pattern in function_patterns)


def fix_dataset_file(file_path):
    """Fix a single sample_sft.json file."""
    print(f"Processing {file_path}...")

    with open(file_path, "r") as f:
        data = json.load(f)

    changes_made = 0

    for item in data:
        if "conversations" not in item:
            continue

        for message in item["conversations"]:
            if "value" not in message or "from" not in message:
                continue

            # Check if this message contains function calls but is marked as 'gpt'
            if message["from"] == "gpt" and contains_function_call(message["value"]):
                message["from"] = "function_call"
                changes_made += 1

    if changes_made > 0:
        # Write back the updated data
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Fixed {changes_made} messages in {file_path}")
    else:
        print(f"  No changes needed in {file_path}")

    return changes_made


def main():
    """Main function to fix all agenttuning datasets."""
    datasets_to_fix = ["agenttuning_alfworld", "agenttuning_db", "agenttuning_kg"]

    total_changes = 0

    for dataset in datasets_to_fix:
        file_path = f"datasets/{dataset}/sample_sft.json"
        if os.path.exists(file_path):
            changes = fix_dataset_file(file_path)
            total_changes += changes
        else:
            print(f"Warning: {file_path} not found")

    print(f"\nTotal changes made: {total_changes}")


if __name__ == "__main__":
    main()
