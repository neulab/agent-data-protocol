import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from schema.trajectory import Trajectory
from schema.version import SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS
from scripts.check_schema_version_bump import (
    extract_schema_version,
    is_schema_impacting_file,
    parse_semver,
)

DATASET_PATH = Path(__file__).parent.parent / "datasets"


def sample_std_paths():
    return sorted(DATASET_PATH.glob("*/sample_std.json"))


def test_schema_version_is_valid_semver():
    assert parse_semver(SCHEMA_VERSION)
    assert SCHEMA_VERSION in SUPPORTED_SCHEMA_VERSIONS
    for version in SUPPORTED_SCHEMA_VERSIONS:
        assert parse_semver(version)


def test_trajectory_defaults_to_current_schema_version():
    trajectory = Trajectory(id="example", content=[])
    assert trajectory.schema_version == SCHEMA_VERSION
    assert trajectory.model_dump()["schema_version"] == SCHEMA_VERSION


def test_trajectory_rejects_unsupported_schema_version():
    with pytest.raises(ValidationError, match="Unsupported schema_version"):
        Trajectory(schema_version="0.0.0", id="example", content=[])


def test_extract_schema_version():
    content = 'from typing import Final\n\nSCHEMA_VERSION: Final = "1.2.3"\n'
    assert extract_schema_version(content) == "1.2.3"


def test_parse_semver_rejects_non_semver_versions():
    with pytest.raises(ValueError, match="MAJOR.MINOR.PATCH"):
        parse_semver("1.0")


def test_schema_impacting_file_detection():
    assert is_schema_impacting_file("schema/trajectory.py")
    assert is_schema_impacting_file("schema/action/api.py")
    assert not is_schema_impacting_file("schema/version.py")
    assert not is_schema_impacting_file("schema/__init__.py")
    assert not is_schema_impacting_file("schema/SCHEMA.md")
    assert not is_schema_impacting_file("datasets/example/schema_raw.py")


@pytest.mark.parametrize("sample_path", sample_std_paths())
def test_sample_std_files_include_current_schema_version(sample_path):
    samples = json.loads(sample_path.read_text())
    assert isinstance(samples, list), f"{sample_path} should contain a list"

    for index, sample in enumerate(samples):
        assert sample.get("schema_version") == SCHEMA_VERSION, (
            f"{sample_path}[{index}] must include 'schema_version': {SCHEMA_VERSION!r}"
        )


def test_schema_version_bump_check_runs_against_main_branch():
    base_ref = None
    for candidate in ("origin/main", "main"):
        result = subprocess.run(
            ["git", "rev-parse", "--verify", candidate],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            base_ref = candidate
            break

    if base_ref is None:
        pytest.skip("No main branch ref is available for the schema version bump check")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_schema_version_bump.py",
            "--base-ref",
            base_ref,
            "--head-ref",
            "HEAD",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
