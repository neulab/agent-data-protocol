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
def test_sample_atif_and_openhands_v0_sft_records_align(subdir):
    """OpenHands v0 SFT samples should preserve normalized ATIF record ids."""
    subdir_path = os.path.join(DATASET_PATH, subdir)
    sample_atif_path = os.path.join(subdir_path, "sample_atif.json")
    sample_sft_path = os.path.join(subdir_path, "sample_sft", "openhands_v0.json")

    if not os.path.exists(sample_atif_path):
        pytest.skip(f"sample_atif.json not found in {subdir_path}")

    assert os.path.exists(sample_sft_path), (
        f"sample_sft/openhands_v0.json not found in {subdir_path}"
    )

    with open(sample_atif_path, "r") as f:
        atif_data = json.load(f)

    with open(sample_sft_path, "r") as f:
        sft_data = json.load(f)

    assert len(atif_data) == len(sft_data), (
        f"Number of samples in ATIF ({len(atif_data)}) and root sft ({len(sft_data)}) "
        f"don't match in {subdir}"
    )

    atif_ids = [sample["trajectory_id"] for sample in atif_data]
    sft_ids = [sample["id"] for sample in sft_data]
    # This test verifies stage alignment only. Some datasets intentionally remain
    # in the broader #218 follow-up for duplicate source IDs within a stage.
    assert atif_ids == sft_ids, f"Sample ids don't match in {subdir}: {atif_ids} vs {sft_ids}"


@pytest.mark.parametrize("subdir", get_subdirectories(DATASET_PATH))
def test_std_to_sft_conversion(subdir):
    """
    Test that sample_sft/openhands_v0.json aligns with ATIF records.

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
    sample_atif_path = os.path.join(subdir_path, "sample_atif.json")
    sample_sft_path = os.path.join(subdir_path, "sample_sft", "openhands_v0.json")

    if not os.path.exists(sample_atif_path):
        pytest.skip(f"sample_atif.json not found in {subdir_path}")

    assert os.path.exists(sample_sft_path), (
        f"sample_sft/openhands_v0.json not found in {subdir_path}"
    )

    # Load the files
    with open(sample_atif_path, "r") as f:
        atif_data = json.load(f)

    with open(sample_sft_path, "r") as f:
        sft_data = json.load(f)

    # Check if the number of samples is the same
    assert len(atif_data) == len(sft_data), (
        f"Number of samples in ATIF ({len(atif_data)}) and sft ({len(sft_data)}) don't match in {subdir}"
    )

    # Check each sample
    for i, (atif_sample, sft_sample) in enumerate(zip(atif_data, sft_data)):
        # Check if IDs match
        assert atif_sample["trajectory_id"] == sft_sample["id"], (
            f"Sample {i} IDs don't match in {subdir}: "
            f"{atif_sample['trajectory_id']} vs {sft_sample['id']}"
        )

        # Check if the SFT sample has the expected structure
        assert "conversations" in sft_sample, (
            f"Sample {i} in {subdir} SFT data missing 'conversations' field"
        )
        assert "system" in sft_sample, f"Sample {i} in {subdir} SFT data missing 'system' field"

        # Check if the number of turns is similar
        # In ATIF format, each turn is an item in the "steps" array
        atif_turns = len(atif_sample["steps"])

        # In SFT format, each turn is an item in the "conversations" array
        sft_turns = len(sft_sample["conversations"])

        # The number of turns might not be exactly the same due to how std_to_sft.py processes the data.
        # For example, system messages might be handled differently. We'll allow some flexibility but
        # ensure the counts are not drastically different.
        if atif_turns > 0 and atif_sample["steps"][0].get("source") == "system":
            atif_turns -= 1

        # ATIF groups an assistant tool call and its observation into one step, while SFT
        # often expands them into separate assistant/user messages. Keep this as a broad
        # sanity check rather than requiring ADP-style one-event-per-turn counts.
        assert sft_turns >= max(1, atif_turns // 2), (
            f"Sample {i} in {subdir} has unexpectedly few SFT turns: "
            f"ATIF has {atif_turns} turns, SFT has {sft_turns} turns"
        )
        assert sft_turns <= max(16, atif_turns * 4), (
            f"Sample {i} in {subdir} has unexpectedly many SFT turns: "
            f"ATIF has {atif_turns} turns, SFT has {sft_turns} turns"
        )
