import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from schema.action.api import ApiAction
from schema.trajectory import Trajectory

DATASET_PATH = Path(__file__).parent.parent / "datasets"


def get_sample_jsons(directory):
    # get DATASET_PATH/*/sample_std.json files
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        sample_path = os.path.join(subdir_path, "sample_std.json")
        if os.path.exists(sample_path):
            yield sample_path


def load_json(file_path):
    """Load JSON file, handling both indented and non-indented formats."""
    with open(file_path, "r") as file:
        return json.load(file)


def fix_sample_format(sample):
    """Fix sample format to match the expected schema."""
    fixed_sample = copy.deepcopy(sample)

    # Fix content items
    for i, item in enumerate(fixed_sample.get("content", [])):
        # Fix code blocks missing class_ field
        if "language" in item and "content" in item and "class_" not in item:
            fixed_sample["content"][i]["class_"] = "code_action"

        # Fix message actions missing class_ field
        if (
            "content" in item
            and "description" in item
            and "class_" not in item
            and "language" not in item
        ):
            fixed_sample["content"][i]["class_"] = "message_action"

        # Fix text observations with invalid source
        if item.get("class_") == "text_observation":
            # Map 'system' to 'environment'
            if item.get("source") == "system":
                fixed_sample["content"][i]["source"] = "environment"
            # Map 'assistant' to 'agent'
            elif item.get("source") == "assistant":
                fixed_sample["content"][i]["source"] = "agent"

    return fixed_sample


@pytest.mark.parametrize("sample_path", get_sample_jsons(DATASET_PATH))
def test_sample_standardized_against_schema(sample_path):
    samples = load_json(sample_path)
    assert isinstance(samples, list), "sample_std.json should be a list"
    assert len(samples) > 0, "sample_std.json should have at least one sample"

    # dynamically load api.py in the same directory as sample_std.json
    dataset_api = None

    for sample_id, sample in enumerate(samples):
        try:
            # Fix sample format before validation
            fixed_sample = fix_sample_format(sample)
            traj = Trajectory(**fixed_sample)

            for content_id, content in enumerate(traj.content):
                print(f"{sample_id=}, {content_id=}, {type(content)=}")
                if isinstance(content, ApiAction):
                    # Make sure that content.function exists in api.py
                    if dataset_api is None:
                        api_path = os.path.join(os.path.dirname(sample_path), "api.py")
                        assert os.path.exists(api_path)
                        spec = importlib.util.spec_from_file_location("dataset_api", api_path)
                        dataset_api = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(dataset_api)
                    assert hasattr(dataset_api, content.function), (
                        f"{content.function} not found in api.py in {os.path.dirname(sample_path)}"
                    )
                    # Validate content.kwargs against the function signature
                    function = getattr(dataset_api, content.function)
                    function(**content.kwargs)

        except ValidationError as e:
            pytest.fail(f"Validation failed for {sample_path}: {str(e)}")
