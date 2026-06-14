import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from schema.atif import ATIF_SCHEMA_VERSION, ATIFTrajectory

DATASET_PATH = Path(__file__).parent.parent / "datasets"


def sample_std_paths():
    return sorted(DATASET_PATH.glob("*/sample_std.json"))


def test_atif_schema_version_is_current():
    assert ATIF_SCHEMA_VERSION == "ATIF-v1.7"


def test_atif_trajectory_defaults_to_current_schema_version():
    trajectory = ATIFTrajectory(trajectory_id="example", steps=[{"step_id": 1, "source": "user"}])
    assert trajectory.schema_version == ATIF_SCHEMA_VERSION
    assert trajectory.model_dump()["schema_version"] == ATIF_SCHEMA_VERSION


def test_atif_trajectory_rejects_unsupported_schema_version():
    with pytest.raises(ValidationError, match="Unsupported ATIF schema_version"):
        ATIFTrajectory(
            schema_version="ATIF-v0",
            trajectory_id="example",
            steps=[{"step_id": 1, "source": "user"}],
        )


@pytest.mark.parametrize("sample_path", sample_std_paths())
def test_sample_std_files_include_atif_schema_version(sample_path):
    samples = json.loads(sample_path.read_text())
    assert isinstance(samples, list), f"{sample_path} should contain a list"

    for index, sample in enumerate(samples):
        trajectory = ATIFTrajectory(**sample)
        assert trajectory.schema_version == ATIF_SCHEMA_VERSION, (
            f"{sample_path}[{index}] must include 'schema_version': {ATIF_SCHEMA_VERSION!r}"
        )
