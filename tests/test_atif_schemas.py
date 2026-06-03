import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from schema.action.code import CodeAction
from schema.atif import (
    ATIF_SCHEMA_VERSION,
    ATIFObservation,
    ATIFTrajectory,
    ObservationResult,
    Step,
    adp_trajectory_to_atif,
    atif_trajectory_to_adp,
    normalize_atif_trajectory,
)
from schema.trajectory import Trajectory

DATASET_PATH = Path(__file__).parent.parent / "datasets"
REPO_ROOT = Path(__file__).parent.parent


def get_sample_atif_jsons(directory):
    for subdir in os.listdir(directory):
        sample_path = Path(directory) / subdir / "sample_atif.json"
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


@pytest.mark.parametrize("sample_path", get_sample_atif_jsons(DATASET_PATH))
def test_sample_atif_and_standardized_records_align(sample_path):
    sample_std_path = sample_path.with_name("sample_std.json")
    if not sample_std_path.exists():
        pytest.skip(f"sample_std.json not found next to {sample_path}")

    atif_data = json.loads(sample_path.read_text())
    std_data = json.loads(sample_std_path.read_text())

    assert [item["trajectory_id"] for item in atif_data] == [item["id"] for item in std_data]


@pytest.mark.parametrize(
    "trajectory",
    [
        {
            "id": "roundtrip-available-apis",
            "available_apis": ["search"],
            "content": [
                {"class_": "text_observation", "content": "Find the answer", "source": "user"},
                {
                    "class_": "api_action",
                    "tool_call_id": "call_1",
                    "function": "search",
                    "kwargs": {"query": "agent data protocol"},
                    "description": "Search first",
                },
                {
                    "class_": "text_observation",
                    "tool_call_id": "call_1",
                    "content": "Result",
                    "source": "environment",
                },
                {"class_": "message_action", "content": "Done"},
            ],
        },
        {
            "id": "roundtrip-tool",
            "content": [
                {"class_": "text_observation", "content": "Find the answer", "source": "user"},
                {
                    "class_": "api_action",
                    "tool_call_id": "call_1",
                    "function": "search",
                    "kwargs": {"query": "agent data protocol"},
                    "description": "Search first",
                },
                {
                    "class_": "text_observation",
                    "tool_call_id": "call_1",
                    "content": "Result",
                    "source": "environment",
                },
                {"class_": "message_action", "content": "Done"},
            ],
        },
        {
            "id": "roundtrip-code",
            "content": [
                {"class_": "text_observation", "content": "Run pwd", "source": "user"},
                {
                    "class_": "code_action",
                    "tool_call_id": "call_1",
                    "language": "bash",
                    "content": "pwd",
                    "description": None,
                },
                {
                    "class_": "text_observation",
                    "tool_call_id": "call_1",
                    "content": "/workspace",
                    "source": "environment",
                },
            ],
        },
        {
            "id": "roundtrip-env-obs",
            "content": [
                {
                    "class_": "text_observation",
                    "content": "System prompt",
                    "source": "environment",
                },
                {"class_": "message_action", "content": "Done"},
            ],
        },
    ],
)
def test_adp_atif_roundtrip_preserves_core_events(trajectory):
    adp = Trajectory(**trajectory)
    atif = adp_trajectory_to_atif(adp)
    normalized_atif = normalize_atif_trajectory(atif)
    roundtripped = atif_trajectory_to_adp(normalized_atif)

    assert roundtripped.id == adp.id
    assert len(roundtripped.content) == len(adp.content)
    assert [item.class_ for item in roundtripped.content] == [item.class_ for item in adp.content]
    assert [getattr(item, "source", None) for item in roundtripped.content] == [
        getattr(item, "source", None) for item in adp.content
    ]
    assert roundtripped.available_apis == adp.available_apis


def test_atif_to_adp_preserves_message_and_standalone_observation():
    atif = ATIFTrajectory(
        trajectory_id="message-with-observation",
        steps=[
            Step(
                step_id=1,
                source="agent",
                message="I inspected the environment.",
                observation=ATIFObservation(results=[ObservationResult(content="env output")]),
            )
        ],
    )

    adp = atif_trajectory_to_adp(atif)

    assert [item.class_ for item in adp.content] == ["message_action", "text_observation"]
    assert adp.content[0].content == "I inspected the environment."
    assert adp.content[1].content == "env output"
    assert getattr(adp.content[1], "source") == "environment"


def test_atif_to_std_script_normalizes_tool_names():
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
        [sys.executable, str(REPO_ROOT / "scripts" / "atif_to_std.py")],
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
    adp = atif_trajectory_to_adp(normalized)
    assert isinstance(adp.content[0], CodeAction)
    assert adp.content[0].language == "bash"
    assert adp.content[0].content == "ls"


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
