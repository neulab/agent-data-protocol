import importlib.util
import sys
from pathlib import Path

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation

DATASET_DIR = Path(__file__).parent.parent / "datasets" / "hybrid-gym"


def load_converter():
    sys.path.insert(0, str(DATASET_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "hybrid_gym_raw_to_standardized", DATASET_DIR / "raw_to_standardized.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(DATASET_DIR))


def test_hybrid_gym_parses_openhands_tool_call_and_result():
    converter = load_converter()
    raw = converter.SchemaRaw(
        id="dep_search_0",
        source_dataset="hybrid-gym/dep_search_502i",
        task_type="dep_search",
        row_index=0,
        messages=[
            {"role": "system", "content": "system prompt omitted"},
            {"role": "user", "content": "Find the dependency."},
            {
                "role": "assistant",
                "content": "I'll inspect it.\n\n<function=execute_bash>\n<parameter=command>pwd</parameter>\n</function>",
            },
            {"role": "user", "content": "EXECUTION RESULT of [execute_bash]:\n/workspace"},
        ],
    )

    trajectory = converter.process_data(raw)

    assert isinstance(trajectory.content[0], TextObservation)
    assert trajectory.content[0].source == "user"
    assert isinstance(trajectory.content[1], CodeAction)
    assert trajectory.content[1].description == "I'll inspect it."
    assert trajectory.content[1].content == "pwd"
    assert trajectory.content[1].tool_call_id == "call_000001"
    assert isinstance(trajectory.content[2], TextObservation)
    assert trajectory.content[2].source == "environment"
    assert trajectory.content[2].tool_call_id == "call_000001"


def test_hybrid_gym_preserves_think_editor_and_finish_actions():
    converter = load_converter()
    assistant_events = converter.convert_assistant_message(
        "<function=think>\n<parameter=thought>Plan</parameter>\n</function>\n"
        "Next edit.\n"
        "<function=str_replace_editor>\n"
        "<parameter=command>view</parameter>\n"
        "<parameter=path>/workspace/project/file.py</parameter>\n"
        "<parameter=view_range>[1, 5]</parameter>\n"
        "</function>\n"
        "<function=finish>\n<parameter=message>Done</parameter>\n</function>"
    )

    assert isinstance(assistant_events[0], ApiAction)
    assert assistant_events[0].function == "think"
    assert assistant_events[0].kwargs == {"thought": "Plan"}
    assert isinstance(assistant_events[1], MessageAction)
    assert assistant_events[1].content == "Next edit."
    assert isinstance(assistant_events[2], ApiAction)
    assert assistant_events[2].function == "str_replace_editor"
    assert assistant_events[2].kwargs["view_range"] == [1, 5]
    assert isinstance(assistant_events[3], MessageAction)
    assert assistant_events[3].content == "<finish> Done </finish>"
