import importlib.util
import sys
from pathlib import Path

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.observation.text import TextObservation

DATASET_DIR = Path(__file__).parent.parent / "datasets" / "SALT-NLP_SWE-chat"


def load_converter_module():
    sys.path.insert(0, str(DATASET_DIR))
    try:
        sys.modules.pop("schema_raw", None)
        spec = importlib.util.spec_from_file_location(
            "swe_chat_raw_to_standardized", DATASET_DIR / "raw_to_standardized.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(DATASET_DIR))


def test_swe_chat_tool_rows_convert_to_adp_actions():
    converter = load_converter_module()
    raw = converter.SchemaRaw(
        session_id="unit-session",
        repo_id="owner/repo",
        agent="Claude Code",
        strategy="auto",
        turns=[
            {
                "turn_number": 0,
                "role": "user",
                "turn_type": "user_prompt",
                "is_conversational": True,
                "content": "Fix the failing tests.",
            },
            {
                "turn_number": 1,
                "role": "assistant",
                "turn_type": "assistant_thinking",
                "content": "I should inspect the test failure first.",
            },
            {
                "turn_number": 2,
                "role": "tool_use",
                "turn_type": "tool_use",
                "tool_name": "Bash",
                "command": "pytest -q",
                "tool_input_json": '{"command": "pytest -q"}',
            },
            {
                "turn_number": 3,
                "role": "tool_result",
                "turn_type": "tool_result",
                "content": "1 failed, 2 passed",
            },
            {
                "turn_number": 4,
                "role": "tool_use",
                "turn_type": "tool_use",
                "tool_name": "Read",
                "file_path": "/workspace/tests/test_example.py",
                "tool_input_json": '{"file_path": "/workspace/tests/test_example.py"}',
            },
            {
                "turn_number": 5,
                "role": "tool_use",
                "turn_type": "tool_use",
                "tool_name": "WebSearch",
                "tool_input_json": '{"query": "pytest assertion introspection"}',
            },
        ],
    )

    trajectory = converter.process_data(raw)

    assert trajectory.id == "unit-session"
    assert isinstance(trajectory.content[0], TextObservation)
    assert trajectory.content[0].source == "user"
    assert isinstance(trajectory.content[1], ApiAction)
    assert trajectory.content[1].function == "think"
    assert isinstance(trajectory.content[2], CodeAction)
    assert trajectory.content[2].language == "bash"
    assert trajectory.content[2].content == "pytest -q"
    assert isinstance(trajectory.content[3], TextObservation)
    assert trajectory.content[3].source == "environment"
    assert isinstance(trajectory.content[4], ApiAction)
    assert trajectory.content[4].function == "str_replace_editor"
    assert trajectory.content[4].kwargs == {
        "command": "view",
        "path": "/workspace/tests/test_example.py",
    }
    assert isinstance(trajectory.content[5], ApiAction)
    assert trajectory.content[5].function == "generic_tool"
    assert trajectory.content[5].kwargs["tool_name"] == "WebSearch"
    assert trajectory.content[5].kwargs["tool_input"] == {"query": "pytest assertion introspection"}
    assert "content" not in trajectory.content[5].kwargs
