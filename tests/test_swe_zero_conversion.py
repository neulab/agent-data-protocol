import importlib.util
import sys
from pathlib import Path

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation

DATASET_DIR = Path(__file__).parent.parent / "datasets" / "nvidia_SWE-Zero-openhands-trajectories"
MODULE_PATH = DATASET_DIR / "raw_to_standardized.py"


def load_converter_module():
    sys.path.insert(0, str(DATASET_DIR))
    try:
        sys.modules.pop("schema_raw", None)
        spec = importlib.util.spec_from_file_location("swe_zero_raw_to_standardized", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(DATASET_DIR))


def test_normalize_tool_observation_strips_only_leading_prefix():
    module = load_converter_module()

    assert module.normalize_tool_observation("OBSERVATION:\nresult") == "result"
    assert (
        module.normalize_tool_observation("before\nOBSERVATION:\ninside output")
        == "before\nOBSERVATION:\ninside output"
    )


def test_swe_zero_tool_calls_convert_to_standardized_events():
    converter = load_converter_module()
    raw = converter.SchemaRaw(
        instance_id="owner__repo-1",
        repo="owner/repo",
        license="MIT",
        trajectory_id="unit-trajectory",
        model_patch="diff --git a/file.py b/file.py",
        dataset="unit/source",
        trajectory=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "content": "I should reason first.",
                "tool_calls": [
                    {
                        "id": "call-think",
                        "type": "function",
                        "function": {"name": "think", "arguments": '{"thought": "inspect"}'},
                    }
                ],
            },
            {"role": "tool", "content": "Your thought has been logged."},
            {
                "role": "assistant",
                "content": "Run tests.",
                "tool_calls": [
                    {
                        "id": "call-bash",
                        "type": "function",
                        "function": {
                            "name": "execute_bash",
                            "arguments": '{"command": "pytest -q"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": "OBSERVATION:\n1 failed"},
            {
                "role": "assistant",
                "content": "Open the file.",
                "tool_calls": [
                    {
                        "id": "call-editor",
                        "type": "function",
                        "function": {
                            "name": "str_replace_editor",
                            "arguments": '{"command": "view", "path": "/workspace/file.py"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": "file contents"},
            {
                "role": "assistant",
                "content": "All done.",
                "tool_calls": [
                    {
                        "id": "call-finish",
                        "type": "function",
                        "function": {"name": "finish", "arguments": '{"message": "Fixed."}'},
                    }
                ],
            },
        ],
    )

    trajectory = converter.process_data(raw)

    assert trajectory.id == "unit-trajectory"
    assert trajectory.details["source_dataset"] == "unit/source"
    assert [type(event) for event in trajectory.content] == [
        TextObservation,
        ApiAction,
        TextObservation,
        CodeAction,
        TextObservation,
        ApiAction,
        TextObservation,
        MessageAction,
    ]
    assert trajectory.content[0].source == "user"
    assert trajectory.content[1].function == "think"
    assert trajectory.content[1].kwargs == {"thought": "inspect"}
    assert trajectory.content[2].source == "environment"
    assert trajectory.content[3].language == "bash"
    assert trajectory.content[3].content == "pytest -q"
    assert trajectory.content[4].content == "1 failed"
    assert trajectory.content[5].function == "str_replace_editor"
    assert trajectory.content[5].kwargs == {"command": "view", "path": "/workspace/file.py"}
    assert trajectory.content[6].source == "environment"
    assert trajectory.content[7].content == "<finish> Fixed. </finish>"
