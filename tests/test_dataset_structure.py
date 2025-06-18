import os
from pathlib import Path

import pytest

DATASET_PATH = Path(__file__).parent.parent / "datasets"


def get_subdirectories(directory):
    ignore_dirs = ["__pycache__"]
    return [
        d
        for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d)) and d not in ignore_dirs
    ]


@pytest.mark.parametrize("subdir", get_subdirectories(DATASET_PATH))
def test_dataset_structure(subdir):
    """Test that each dataset has the required files."""
    subdir_path = os.path.join(DATASET_PATH, subdir)

    # All datasets should have sample_raw.json
    sample_raw_path = os.path.join(subdir_path, "sample_raw.json")
    assert os.path.exists(sample_raw_path), f"sample_raw.json not found in {subdir_path}"

    # If raw_to_standardized.py exists, the dataset should have sample_std.json
    raw_to_std_path = os.path.join(subdir_path, "raw_to_standardized.py")
    sample_std_path = os.path.join(subdir_path, "sample_std.json")

    if os.path.exists(raw_to_std_path):
        assert os.path.exists(sample_std_path), (
            f"raw_to_standardized.py exists but sample_std.json not found in {subdir_path}"
        )

    # If sample_std.json exists, then sample_sft.json should exist
    if os.path.exists(sample_std_path):
        sample_sft_path = os.path.join(subdir_path, "sample_sft.json")
        assert os.path.exists(sample_sft_path), (
            f"sample_std.json exists but sample_sft.json not found in {subdir_path}"
        )

    # Check for other JSON files that shouldn't be there
    allowed_jsons = [
        "sample_raw.json",
        "sample_std.json",
        "sample_sft.json",
        "generated_thoughts.json",
    ]
    for file in os.listdir(subdir_path):
        if file.endswith(".json") and file not in allowed_jsons:
            # Special case for androidcontrol which has a nested directory structure
            if subdir == "androidcontrol" and (
                file == "splits.json" or file == "trajectories.json"
            ):
                continue
            pytest.fail(f"Unexpected JSON file found: {file} in {subdir_path}")
