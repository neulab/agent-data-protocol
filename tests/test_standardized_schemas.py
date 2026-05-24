import importlib.util
import inspect
import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from schema.action.api import ApiAction
from schema.trajectory import Trajectory

DATASET_PATH = Path(__file__).parent.parent / "datasets"
NUMERIC_DETAIL_KEYS = {"reward", "score", "step"}
NUMERIC_DETAIL_KEY_PREFIXES = {"pred_passes_"}
NUMERIC_DETAIL_KEY_SUFFIXES = {
    "_correct",
    "_count",
    "_index",
    "_number",
    "_percentage",
    "_percent",
    "_rate",
    "_reward",
    "_score",
    "_success",
}


def is_numeric_detail_key(key):
    key = key.lower()
    return (
        key in NUMERIC_DETAIL_KEYS
        or any(key.startswith(prefix) for prefix in NUMERIC_DETAIL_KEY_PREFIXES)
        or any(key.endswith(suffix) for suffix in NUMERIC_DETAIL_KEY_SUFFIXES)
    )


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


def test_numeric_detail_key_detection():
    assert is_numeric_detail_key("tool_call_count")
    assert is_numeric_detail_key("agent_percentage")
    assert is_numeric_detail_key("session_success")
    assert is_numeric_detail_key("rollout_number")
    assert is_numeric_detail_key("source_index")
    assert is_numeric_detail_key("reward")
    assert is_numeric_detail_key("gen_tests_correct")
    assert is_numeric_detail_key("pred_passes_gen_tests")

    assert not is_numeric_detail_key("source_id")
    assert not is_numeric_detail_key("timestamp")
    assert not is_numeric_detail_key("version")
    assert not is_numeric_detail_key("answer")


@pytest.mark.parametrize("sample_path", get_sample_jsons(DATASET_PATH))
def test_sample_standardized_against_schema(sample_path):
    samples = load_json(sample_path)
    assert isinstance(samples, list), "sample_std.json should be a list"
    assert len(samples) > 0, "sample_std.json should have at least one sample"

    # dynamically load api.py in the same directory as sample_std.json
    dataset_api = None
    api_function_names = None

    def load_dataset_api():
        nonlocal dataset_api, api_function_names
        if dataset_api is None:
            api_path = os.path.join(os.path.dirname(sample_path), "api.py")
            assert os.path.exists(api_path)
            spec = importlib.util.spec_from_file_location("dataset_api", api_path)
            dataset_api = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dataset_api)
            api_function_names = {
                name for name, _ in inspect.getmembers(dataset_api, inspect.isfunction)
            }
        return dataset_api, api_function_names

    for sample_id, sample in enumerate(samples):
        try:
            traj = Trajectory(**sample)
            assert "available_apis" not in traj.details, (
                f"available_apis must be a top-level Trajectory field, not details metadata, "
                f"in {sample_path} sample {sample_id}"
            )
            stringified_numeric_details = {
                key: value
                for key, value in traj.details.items()
                if is_numeric_detail_key(key) and isinstance(value, str)
            }
            assert not stringified_numeric_details, (
                f"Numeric details must be stored as native JSON numbers in {sample_path} "
                f"sample {sample_id}: {stringified_numeric_details}"
            )
            if traj.available_apis is not None:
                available_apis = traj.available_apis
                _, api_function_names = load_dataset_api()
                missing_available_apis = sorted(set(available_apis) - api_function_names)
                assert not missing_available_apis, (
                    f"available_apis contains functions not found in api.py in "
                    f"{os.path.dirname(sample_path)}: {missing_available_apis}"
                )
                used_apis = {
                    content.function for content in traj.content if isinstance(content, ApiAction)
                }
                missing_used_apis = sorted(used_apis - set(available_apis))
                assert not missing_used_apis, (
                    f"ApiAction functions are missing from available_apis in {sample_path} "
                    f"sample {sample_id}: {missing_used_apis}"
                )

            for content_id, content in enumerate(traj.content):
                print(f"{sample_id=}, {content_id=}, {type(content)=}")
                if isinstance(content, ApiAction):
                    # Make sure that content.function exists in api.py
                    dataset_api, _ = load_dataset_api()
                    assert hasattr(dataset_api, content.function), (
                        f"{content.function} not found in api.py in {os.path.dirname(sample_path)}"
                    )
                    # Validate content.kwargs against the function signature
                    function = getattr(dataset_api, content.function)
                    function(**content.kwargs)

        except ValidationError as e:
            pytest.fail(f"Validation failed for {sample_path}: {str(e)}")
