import glob
import json
import os
import unittest


class TestDatasetsFromParameter(unittest.TestCase):
    """Test that all datasets use the correct 'from' parameter for function calls."""

    def test_all_datasets_function_call_from_parameter(self):
        """Test that all datasets use 'from': 'function_call' for function calls."""
        # Get all sample_sft.json files in the datasets directory
        datasets_dir = os.path.join(os.path.dirname(__file__), "../datasets")
        sample_sft_files = glob.glob(f"{datasets_dir}/**/sample_sft.json", recursive=True)

        self.assertGreater(len(sample_sft_files), 0, "No sample_sft.json files found")

        # Track datasets that need to be fixed
        datasets_to_fix = set()

        # Check each sample_sft.json file
        for file_path in sample_sft_files:
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    self.fail(f"Failed to parse JSON in {file_path}")

            # Check each conversation in the dataset
            for item in data:
                if "conversations" not in item:
                    continue

                for message in item["conversations"]:
                    # Check if this is a function call by looking for function patterns
                    if "value" not in message:
                        continue

                    value = message["value"]
                    is_function_call = any(
                        [
                            "<function=" in value,
                            "<function_calls>" in value,
                            "<invoke name=" in value,
                        ]
                    )

                    if is_function_call and message["from"] != "function_call":
                        # Add this dataset to the list of datasets that need to be fixed
                        relative_path = os.path.relpath(file_path, datasets_dir)
                        dataset_name = relative_path.split("/")[0]
                        datasets_to_fix.add((dataset_name, message["from"]))

        # Print the datasets that need to be fixed and fail the test
        if datasets_to_fix:
            error_message = "\nDatasets that need to be fixed:\n"
            for dataset_name, from_value in sorted(datasets_to_fix):
                error_message += f"  - {dataset_name}: 'from': '{from_value}' should be 'from': 'function_call'\n"

            # Fail the test with a clear error message
            self.fail(
                f"{error_message}\nFound {len(datasets_to_fix)} datasets that need to be fixed. "
                f"Please update the datasets or modify std_to_sft.py to ensure consistent use of 'from': 'function_call'."
            )


if __name__ == "__main__":
    unittest.main()
