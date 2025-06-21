import json
import os
from pathlib import Path

import pytest

DATASET_PATH = Path(__file__).parent.parent / "datasets"

# Datasets that are not completely finished (not documented in DATASETS.md)
INCOMPLETE_DATASETS = [
    "android_in_the_wild",
    "androidcontrol",
    "eto",
    "go-browse-wa",
    "llava_plus",
    "mind2web",
    "omniact",
    "screenagent",
    "turkingbench",
    "webarena_successful",
    "weblinx",
    "wonderbread",
]


def get_subdirectories(directory):
    ignore_dirs = ["__pycache__"]
    return [
        d
        for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d)) and d not in ignore_dirs
    ]


@pytest.mark.parametrize("subdir", get_subdirectories(DATASET_PATH))
def test_std_to_sft_conversion(subdir):
    """
    Test that sample_sft.json is correctly generated from sample_std.json.

    Checks:
    1. Both files exist
    2. The number of samples is the same
    3. Each sample has the expected structure
    4. The number of turns in each sample is similar
    """
    # Skip incomplete datasets
    if subdir in INCOMPLETE_DATASETS:
        pytest.skip(f"Skipping incomplete dataset: {subdir}")

    subdir_path = os.path.join(DATASET_PATH, subdir)

    # Check if both files exist
    sample_std_path = os.path.join(subdir_path, "sample_std.json")
    sample_sft_path = os.path.join(subdir_path, "sample_sft.json")

    if not os.path.exists(sample_std_path):
        pytest.skip(f"sample_std.json not found in {subdir_path}")

    assert os.path.exists(sample_sft_path), f"sample_sft.json not found in {subdir_path}"

    # Load the files
    with open(sample_std_path, "r") as f:
        std_data = json.load(f)

    with open(sample_sft_path, "r") as f:
        sft_data = json.load(f)

    # Check if the number of samples is the same
    assert len(std_data) == len(sft_data), (
        f"Number of samples in std ({len(std_data)}) and sft ({len(sft_data)}) don't match in {subdir}"
    )

    # Check each sample
    for i, (std_sample, sft_sample) in enumerate(zip(std_data, sft_data)):
        # Check if IDs match
        assert std_sample["id"] == sft_sample["id"], (
            f"Sample {i} IDs don't match in {subdir}: {std_sample['id']} vs {sft_sample['id']}"
        )

        # Check if the SFT sample has the expected structure
        assert "conversations" in sft_sample, (
            f"Sample {i} in {subdir} SFT data missing 'conversations' field"
        )
        assert "system" in sft_sample, f"Sample {i} in {subdir} SFT data missing 'system' field"

        # Check if the number of turns is similar
        # In STD format, each turn is an item in the "content" array
        std_turns = len(std_sample["content"])

        # In SFT format, each turn is an item in the "conversations" array
        sft_turns = len(sft_sample["conversations"])

        # The number of turns might not be exactly the same due to how std_to_sft.py processes the data
        # For example, system messages might be handled differently
        # But they should be reasonably close
        # We'll allow for some flexibility but ensure they're not drastically different

        # Skip system message in std_turns count if it exists
        if (
            std_turns > 0
            and "source" in std_sample["content"][0]
            and std_sample["content"][0]["source"] == "system"
        ):
            std_turns -= 1

        # Check if the number of turns is within a reasonable range
        # Allow for some difference but not too much
        max_diff = max(2, std_turns * 0.3)  # Allow either 2 turns difference or 30% difference

        assert abs(std_turns - sft_turns) <= max_diff, (
            f"Sample {i} in {subdir} has too different number of turns: "
            f"STD has {std_turns} turns, SFT has {sft_turns} turns"
        )
