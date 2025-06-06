#!/usr/bin/env python3
"""
Fix source fields in standardized datasets to match schema requirements.

The schema only allows 'user', 'agent', or 'environment' as source values,
but some datasets have 'system' or 'os' which need to be mapped to 'environment'.
"""

import json
import os


def fix_source_fields(file_path, source_mapping):
    """Fix source fields in a JSON file according to the mapping."""
    print(f"Processing {file_path}...")

    with open(file_path, "r") as f:
        data = json.load(f)

    changes_made = 0

    for trajectory in data:
        for content in trajectory["content"]:
            if isinstance(content, dict) and "source" in content:
                old_source = content["source"]
                if old_source in source_mapping:
                    content["source"] = source_mapping[old_source]
                    changes_made += 1
                    print(f"  Changed source '{old_source}' -> '{source_mapping[old_source]}'")

    if changes_made > 0:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Made {changes_made} changes and saved file.")
    else:
        print("  No changes needed.")

    return changes_made


def main():
    # Define the files and their source mappings
    files_to_fix = [
        {
            "path": "datasets/agenttuning_webshop/sample_std.json",
            "mapping": {"system": "environment"},
        },
        {"path": "datasets/agenttuning_os/sample_std.json", "mapping": {"os": "environment"}},
    ]

    total_changes = 0

    for file_info in files_to_fix:
        file_path = file_info["path"]
        if os.path.exists(file_path):
            changes = fix_source_fields(file_path, file_info["mapping"])
            total_changes += changes
        else:
            print(f"Warning: File {file_path} not found!")

    print(f"\nTotal changes made: {total_changes}")

    if total_changes > 0:
        print("Files have been updated. Please run the tests to verify the fixes.")


if __name__ == "__main__":
    main()
