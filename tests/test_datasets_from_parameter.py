import glob
import json
import os

import pytest

FUNCTION_CALL_PATTERNS = ("<function=", "<function_calls>", "<invoke name=")


def test_all_openhands_v0_datasets_function_call_from_parameter():
    """Test that OpenHands v0 SFT samples use 'from': 'function_call' for function calls."""
    # Get all OpenHands v0 SFT sample files in the datasets directory.
    datasets_dir = os.path.join(os.path.dirname(__file__), "../datasets")
    sample_sft_files = glob.glob(f"{datasets_dir}/**/sample_sft/openhands_v0.json", recursive=True)

    assert len(sample_sft_files) > 0, "No sample_sft/openhands_v0.json files found"

    # Track datasets that need to be fixed
    datasets_to_fix = set()

    # Check each OpenHands v0 SFT sample file.
    for file_path in sample_sft_files:
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                pytest.fail(f"Failed to parse JSON in {file_path}")

        # Check each conversation in the dataset
        for item in data:
            if "conversations" not in item:
                continue

            for message in item["conversations"]:
                # Check if this is a function call by looking for function patterns
                if "value" not in message:
                    continue

                value = message["value"]
                is_function_call = any(pattern in value for pattern in FUNCTION_CALL_PATTERNS)

                if is_function_call and message["from"] != "function_call":
                    # Add this dataset to the list of datasets that need to be fixed
                    relative_path = os.path.relpath(file_path, datasets_dir)
                    dataset_name = relative_path.split("/")[0]
                    datasets_to_fix.add((dataset_name, message["from"]))

    # Print the datasets that need to be fixed and fail the test
    if datasets_to_fix:
        error_message = "\nDatasets that need to be fixed:\n"
        for dataset_name, from_value in sorted(datasets_to_fix):
            error_message += (
                f"  - {dataset_name}: 'from': '{from_value}' should be 'from': 'function_call'\n"
            )

        # Fail the test with a clear error message
        pytest.fail(
            f"{error_message}\nFound {len(datasets_to_fix)} datasets that need to be fixed. "
            f"Please update the datasets or modify std_to_sft.py to ensure consistent use of 'from': 'function_call'."
        )


def test_all_datasets_function_call_role_contains_function_call():
    """Test that 'from': 'function_call' is reserved for function-call messages."""
    datasets_dir = os.path.join(os.path.dirname(__file__), "../datasets")
    sample_sft_files = glob.glob(f"{datasets_dir}/**/sample_sft/openhands_v0.json", recursive=True)

    assert len(sample_sft_files) > 0, "No sample_sft/openhands_v0.json files found"

    datasets_to_fix = set()

    for file_path in sample_sft_files:
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                pytest.fail(f"Failed to parse JSON in {file_path}")

        for item in data:
            if "conversations" not in item:
                continue

            for message in item["conversations"]:
                value = message.get("value", "")
                has_function_call = any(pattern in value for pattern in FUNCTION_CALL_PATTERNS)

                if message.get("from") == "function_call" and not has_function_call:
                    relative_path = os.path.relpath(file_path, datasets_dir)
                    dataset_name = relative_path.split("/")[0]
                    datasets_to_fix.add(dataset_name)

    if datasets_to_fix:
        error_message = "\nDatasets that need to be fixed:\n"
        for dataset_name in sorted(datasets_to_fix):
            error_message += (
                f"  - {dataset_name}: 'from': 'function_call' messages must contain "
                "function-call syntax\n"
            )

        pytest.fail(
            f"{error_message}\nFound {len(datasets_to_fix)} datasets that need to be fixed. "
            "Please update the dataset converters and regenerate sample_sft/openhands_v0.json."
        )
