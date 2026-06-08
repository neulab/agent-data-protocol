import json
import os
from pathlib import Path

import pytest

from schema.dataset_metadata import DatasetMetadata

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
        "put normalization in raw_to_atif.py or atif_to_std.py, then use a shared "
        "converter under agents/ instead"
    )

    dataset_api_path = os.path.join(subdir_path, "api.py")
    assert not os.path.exists(dataset_api_path), (
        f"Dataset-local api.py is not allowed in {subdir_path}; "
        "define custom tools in metadata.json instead"
    )

    metadata_path = Path(subdir_path) / "metadata.json"
    assert metadata_path.exists(), f"metadata.json not found in {subdir_path}"
    metadata_data = json.loads(metadata_path.read_text())
    DatasetMetadata.model_validate(metadata_data)
    assert (
        metadata_path.read_text() == json.dumps(metadata_data, indent=2, ensure_ascii=False) + "\n"
    ), f"metadata.json is not formatted with 2-space indentation: {metadata_path}"

    # All datasets should have sample_raw.json
    sample_raw_path = os.path.join(subdir_path, "sample_raw.json")
    assert os.path.exists(sample_raw_path), f"sample_raw.json not found in {subdir_path}"

    # If raw_to_standardized.py exists, the dataset should have standardized and ATIF samples.
    raw_to_std_path = os.path.join(subdir_path, "raw_to_standardized.py")
    raw_to_atif_path = os.path.join(subdir_path, "raw_to_atif.py")
    atif_to_std_path = os.path.join(subdir_path, "atif_to_std.py")
    sample_std_path = os.path.join(subdir_path, "sample_std.json")
    sample_atif_path = os.path.join(subdir_path, "sample_atif.json")

    if os.path.exists(raw_to_std_path):
        assert os.path.exists(raw_to_atif_path), (
            f"raw_to_standardized.py exists but raw_to_atif.py not found in {subdir_path}"
        )
        assert os.path.exists(atif_to_std_path), (
            f"raw_to_standardized.py exists but atif_to_std.py not found in {subdir_path}"
        )
        assert os.path.exists(sample_std_path), (
            f"raw_to_standardized.py exists but sample_std.json not found in {subdir_path}"
        )
        assert os.path.exists(sample_atif_path), (
            f"raw_to_standardized.py exists but sample_atif.json not found in {subdir_path}"
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
        "metadata.json",
        "sample_raw.json",
        "sample_atif.json",
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
