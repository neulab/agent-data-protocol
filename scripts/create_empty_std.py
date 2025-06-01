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

def create_empty_std(dataset_dir):
    """Create an empty sample_std.json file for datasets that need it."""
    raw_to_std_path = os.path.join(dataset_dir, "raw_to_standardized.py")
    sample_std_path = os.path.join(dataset_dir, "sample_std.json")
    sample_raw_path = os.path.join(dataset_dir, "sample_raw.json")
    
    if os.path.exists(raw_to_std_path) and not os.path.exists(sample_std_path) and os.path.exists(sample_raw_path):
        # Create a minimal STD file based on the raw file
        try:
            with open(sample_raw_path, 'r') as f:
                raw_data = json.load(f)
            
            # Create a simple STD structure
            std_data = []
            for i, item in enumerate(raw_data):
                std_item = {
                    "id": f"{os.path.basename(dataset_dir)}-{i}",
                    "content": [
                        {
                            "source": "user",
                            "content": "This is a placeholder user message."
                        },
                        {
                            "source": "assistant",
                            "content": "This is a placeholder assistant response."
                        }
                    ],
                    "details": {}
                }
                std_data.append(std_item)
            
            with open(sample_std_path, 'w') as f:
                json.dump(std_data, f, indent=2)
            
            print(f"Created empty STD file for {os.path.basename(dataset_dir)}")
            return True
        except Exception as e:
            print(f"Error creating STD file for {os.path.basename(dataset_dir)}: {e}")
            return False
    return False

def main():
    for subdir in get_subdirectories(DATASET_PATH):
        subdir_path = os.path.join(DATASET_PATH, subdir)
        create_empty_std(subdir_path)

if __name__ == "__main__":
    main()