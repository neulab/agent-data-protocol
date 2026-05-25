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

    dataset_sft_converter_path = os.path.join(subdir_path, "std_to_sft.py")
    assert not os.path.exists(dataset_sft_converter_path), (
        f"Dataset-local std_to_sft.py is not allowed in {subdir_path}; "
        "use a shared converter under agents/ instead"
    )

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

    # If sample_std.json exists, then an OpenHands v0 SFT sample should exist.
    if os.path.exists(sample_std_path):
        sample_sft_dir = os.path.join(subdir_path, "sample_sft")
        openhands_v0_sft_path = os.path.join(sample_sft_dir, "openhands_v0.json")
        assert os.path.isdir(sample_sft_dir), (
            f"sample_std.json exists but sample_sft directory not found in {subdir_path}"
        )
        assert os.path.exists(openhands_v0_sft_path), (
            f"sample_std.json exists but sample_sft/openhands_v0.json not found in {subdir_path}"
        )

    # Check for other JSON files that shouldn't be there
    allowed_jsons = [
        "sample_raw.json",
        "sample_std.json",
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

    sample_sft_dir = os.path.join(subdir_path, "sample_sft")
    if os.path.isdir(sample_sft_dir):
        for file in os.listdir(sample_sft_dir):
            if file.endswith(".json") and file.startswith("sample_sft_"):
                pytest.fail(
                    f"Legacy SFT sample filename found: {file} in {sample_sft_dir}. "
                    "Use sample_sft/{agent_name}.json instead."
                )
