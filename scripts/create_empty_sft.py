import json
import os
import sys
from pathlib import Path

DATASET_PATH = Path(__file__).parent.parent / "datasets"

def get_subdirectories(directory):
    ignore_dirs = ["__pycache__"]
    return [
        d
        for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d)) and d not in ignore_dirs
    ]

def create_empty_sft(dataset_dir):
    """Create an empty sample_sft.json file for datasets that need it."""
    sample_std_path = os.path.join(dataset_dir, "sample_std.json")
    sample_sft_path = os.path.join(dataset_dir, "sample_sft.json")
    
    if os.path.exists(sample_std_path) and not os.path.exists(sample_sft_path):
        # Create a minimal SFT file based on the STD file
        try:
            with open(sample_std_path, 'r') as f:
                std_data = json.load(f)
            
            # Create a simple SFT structure
            sft_data = []
            for item in std_data:
                sft_item = {
                    "id": item.get("id", "unknown"),
                    "conversations": [
                        {"from": "human", "value": "This is a placeholder conversation."},
                        {"from": "gpt", "value": "This is a placeholder response."}
                    ],
                    "system": "You are a helpful AI assistant."
                }
                sft_data.append(sft_item)
            
            with open(sample_sft_path, 'w') as f:
                json.dump(sft_data, f, indent=2)
            
            print(f"Created empty SFT file for {os.path.basename(dataset_dir)}")
            return True
        except Exception as e:
            print(f"Error creating SFT file for {os.path.basename(dataset_dir)}: {e}")
            return False
    return False

def main():
    for subdir in get_subdirectories(DATASET_PATH):
        subdir_path = os.path.join(DATASET_PATH, subdir)
        create_empty_sft(subdir_path)

if __name__ == "__main__":
    main()