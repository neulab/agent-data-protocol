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
    # get DATASET_PATH/*/sample.json files
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        sample_path = os.path.join(subdir_path, "sample.json")
        if os.path.exists(sample_path):
            yield sample_path


def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


@pytest.mark.parametrize("sample_path", get_sample_jsons(DATASET_PATH))
def test_sample_raw_against_schema(sample_path):
    samples = load_json(sample_path)
    assert isinstance(samples, list), "sample.json should be a list"

    # dynamically load api.py in the same directory as sample.json
    dataset_api = None

    for sample in samples:
        try:
            traj = Trajectory(**sample)
            for content in traj.content:
                if isinstance(content, ApiAction):
                    # Make sure that content.function exists in api.py
                    if dataset_api is None:
                        api_path = os.path.join(os.path.dirname(sample_path), "api.py")
                        assert os.path.exists(api_path)
                        spec = importlib.util.spec_from_file_location("dataset_api", api_path)
                        dataset_api = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(dataset_api)
                    assert hasattr(dataset_api, content.function), (
                        f"{content.function} not found in api.py"
                        f" in {os.path.dirname(sample_path)}"
                    )
                    # Validate content.kwargs against the function signature
                    function = getattr(dataset_api, content.function)
                    function(**content.kwargs)

        except ValidationError as e:
            pytest.fail(f"Validation failed for {sample_path}: {str(e)}")
