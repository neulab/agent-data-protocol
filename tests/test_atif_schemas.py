import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from schema.atif import (
    ATIF_SCHEMA_VERSION,
    ATIFTrajectory,
)

DATASET_PATH = Path(__file__).parent.parent / "datasets"
REPO_ROOT = Path(__file__).parent.parent


def get_sample_atif_jsons(directory):
    for subdir in os.listdir(directory):
        sample_path = Path(directory) / subdir / "sample_atif.json"
        if sample_path.exists():
            yield sample_path


def get_sample_std_jsons(directory):
    for subdir in os.listdir(directory):
        sample_path = Path(directory) / subdir / "sample_std.json"
        if sample_path.exists():
            yield sample_path


def test_atif_schema_rejects_unknown_fields_and_bad_links():
    with pytest.raises(ValidationError):
        ATIFTrajectory(
            schema_version=ATIF_SCHEMA_VERSION,
            trajectory_id="bad-extra",
            agent={"name": "test", "version": "0"},
            steps=[{"step_id": 1, "source": "user", "message": "hello", "extra_field": True}],
        )

    with pytest.raises(ValidationError):
        ATIFTrajectory(
            schema_version=ATIF_SCHEMA_VERSION,
            trajectory_id="bad-link",
            agent={"name": "test", "version": "0"},
            steps=[
                {
                    "step_id": 1,
                    "source": "agent",
                    "message": "",
                    "tool_calls": [
                        {
                            "tool_call_id": "call_1",
                            "function_name": "search",
                            "arguments": {"query": "adp"},
                        }
                    ],
                    "observation": {
                        "results": [
                            {"source_call_id": "call_2", "content": "result"},
                        ]
                    },
                }
            ],
        )


@pytest.mark.parametrize("sample_path", get_sample_atif_jsons(DATASET_PATH))
def test_sample_atif_schema(sample_path):
    data = json.loads(sample_path.read_text())
    assert data, f"{sample_path} should contain at least one trajectory"
    for trajectory in data:
        parsed = ATIFTrajectory(**trajectory)
        assert parsed.schema_version == ATIF_SCHEMA_VERSION


@pytest.mark.parametrize("sample_path", get_sample_std_jsons(DATASET_PATH))
def test_sample_std_schema(sample_path):
    data = json.loads(sample_path.read_text())
    assert data, f"{sample_path} should contain at least one trajectory"
    for trajectory in data:
        parsed = ATIFTrajectory(**trajectory)
        assert parsed.schema_version == ATIF_SCHEMA_VERSION


@pytest.mark.parametrize("sample_path", get_sample_atif_jsons(DATASET_PATH))
def test_sample_atif_and_standardized_records_align(sample_path):
    sample_std_path = sample_path.with_name("sample_std.json")
    if not sample_std_path.exists():
        pytest.skip(f"sample_std.json not found next to {sample_path}")

    atif_data = json.loads(sample_path.read_text())
    std_data = json.loads(sample_std_path.read_text())

    assert [item["trajectory_id"] for item in atif_data] == [
        item["trajectory_id"] for item in std_data
    ]


def test_raw_to_atif_common_does_not_depend_on_adp_standardization():
    source = (REPO_ROOT / "scripts" / "raw_to_atif_common.py").read_text()
    assert "raw_to_standardized" not in source
    assert "schema.trajectory" not in source


def test_raw_to_atif_preserves_raw_tool_name_before_normalization():
    raw_sample = json.loads((DATASET_PATH / "codeactinstruct" / "sample_raw.json").read_text())[0]
    process = subprocess.run(
        [sys.executable, str(DATASET_PATH / "codeactinstruct" / "raw_to_atif.py")],
        input=json.dumps(raw_sample) + "\n",
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        check=True,
    )
    atif = ATIFTrajectory(**json.loads(process.stdout))
    raw_tool_names = [
        tool_call.function_name for step in atif.steps for tool_call in (step.tool_calls or [])
    ]
    assert "execute" in raw_tool_names
    assert "execute_ipython_cell" not in raw_tool_names


@pytest.mark.parametrize("dataset_name", ["codeactinstruct", "coderforge_preview"])
def test_dataset_atif_to_std_script_normalizes_tool_names(dataset_name):
    atif = ATIFTrajectory(
        trajectory_id="normalize-shell",
        agent={"name": "test", "version": "0"},
        steps=[
            {
                "step_id": 1,
                "source": "agent",
                "message": "",
                "tool_calls": [
                    {
                        "tool_call_id": "call_1",
                        "function_name": "shell",
                        "arguments": {"code": "ls"},
                        "extra": {"adp_class": "code_action", "language": "shell"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "call_1", "content": "file"}]},
            }
        ],
    )
    process = subprocess.run(
        [sys.executable, str(DATASET_PATH / dataset_name / "atif_to_std.py")],
        input=atif.model_dump_json(exclude_none=True) + "\n",
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        check=True,
    )
    normalized = ATIFTrajectory(**json.loads(process.stdout))
    assert normalized.steps[0].tool_calls[0].function_name == "execute_bash"
    assert normalized.steps[0].tool_calls[0].arguments == {"command": "ls"}
    assert normalized.steps[0].tool_calls[0].extra == {
        "adp_class": "code_action",
        "language": "bash",
    }


def test_openhands_v0_std_to_sft_accepts_atif_input():
    atif_sample = json.loads((DATASET_PATH / "codeactinstruct" / "sample_atif.json").read_text())[0]
    env = os.environ.copy()
    env["MY_DATASET"] = "codeactinstruct"
    process = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "agents" / "openhands_v0" / "std_to_sft.py"),
            "--is_web",
            "no",
            "--api_env",
            "execute_ipython_cell",
        ],
        input=json.dumps(atif_sample) + "\n",
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    converted = json.loads(process.stdout)
    conversations = converted["conversations"]
    assert converted["id"] == atif_sample["trajectory_id"]
    assert len(conversations) >= 3
    assert all(set(message) == {"role", "content"} for message in conversations)
    assert any(
        message["role"] == "assistant" and "<function=execute_ipython_cell>" in message["content"]
        for message in conversations
    )
