import os
import json
import pytest
import importlib.util
from pathlib import Path
from pydantic import ValidationError

DATASET_PATH = Path(__file__).parent.parent / "datasets"


def get_subdirectories(directory):
    ignore_dirs = ["__pycache__"]
    return [
        d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d)) and d not in ignore_dirs
    ]


def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def load_schema_class(schema_path):
    spec = importlib.util.spec_from_file_location("schema_raw", schema_path)
    schema_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema_module)
    return schema_module.SchemaRaw


@pytest.mark.parametrize("subdir", get_subdirectories(DATASET_PATH))
def test_sample_raw_against_schema(subdir):
    subdir_path = os.path.join(DATASET_PATH, subdir)

    schema_path = os.path.join(subdir_path, "schema_raw.py")
    sample_path = os.path.join(subdir_path, "sample_raw.json")

    assert os.path.exists(
        schema_path
    ), f"schema_raw.py not found in {subdir_path}"
    assert os.path.exists(
        sample_path
    ), f"sample_raw.json not found in {subdir_path}"

    SchemaRaw = load_schema_class(schema_path)
    samples = load_json(sample_path)

    assert isinstance(samples, list), "sample_raw.json should be a list"

    for sample in samples:
        try:
            if isinstance(sample, dict):
                SchemaRaw(**sample)
            elif isinstance(sample, list):
                SchemaRaw(root=sample)
            else:
                pytest.fail(f"Invalid sample type for {subdir}")
        except ValidationError as e:
            pytest.fail(f"Validation failed for {subdir}: {str(e)}")
