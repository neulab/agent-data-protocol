import json
from pathlib import Path

import pytest

from schema.dataset_metadata import load_dataset_metadata

DATASET_PATH = Path(__file__).parent.parent / "datasets"


def dataset_dirs():
    return sorted(path for path in DATASET_PATH.iterdir() if path.is_dir())


@pytest.mark.parametrize("dataset_dir", dataset_dirs())
def test_metadata_sample_expectations(dataset_dir):
    metadata = load_dataset_metadata(dataset_dir.name, required=True)
    expectations = metadata.sample_expectations
    if not any(
        value is not None
        for value in [
            expectations.min_std_steps,
            expectations.min_std_tool_calls,
            expectations.min_sdk_messages,
        ]
    ):
        return

    std_rows = json.loads((dataset_dir / "sample_std.json").read_text())
    sdk_rows = json.loads((dataset_dir / "sample_sft" / "openhands_sdk.json").read_text())
    std_steps = sum(len(row.get("steps", [])) for row in std_rows)
    std_tool_calls = sum(
        len(step.get("tool_calls") or []) for row in std_rows for step in row.get("steps", [])
    )
    sdk_messages = sum(len(row.get("messages", [])) for row in sdk_rows)

    if expectations.min_std_steps is not None:
        assert std_steps >= expectations.min_std_steps, (
            f"{dataset_dir.name} sample_std.json has too few steps: "
            f"{std_steps} < {expectations.min_std_steps}"
        )
    if expectations.min_std_tool_calls is not None:
        assert std_tool_calls >= expectations.min_std_tool_calls, (
            f"{dataset_dir.name} sample_std.json has too few tool calls: "
            f"{std_tool_calls} < {expectations.min_std_tool_calls}"
        )
    if expectations.min_sdk_messages is not None:
        assert sdk_messages >= expectations.min_sdk_messages, (
            f"{dataset_dir.name} sample_sft/openhands_sdk.json has too few messages: "
            f"{sdk_messages} < {expectations.min_sdk_messages}"
        )
