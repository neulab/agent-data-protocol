import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
DATASET_PATH = ROOT / "datasets"
AGENTS_PATH = ROOT / "agents"

NON_SYSTEM_PROMPT_PATTERNS = [
    "You are OpenHands agent",
    "<ROLE>",
    "You are a UI Assistant",
    "You are web shopping",
    "Interact with a household to solve a task",
    "You are an agent for science world",
    "The following pre-defined functions",
    "provides the following pre-defined functions",
    "Available functions in ",
    "The following functions are ",
    "supports the following pre-defined functions",
    "Below is a list of functions",
    "The toolkit for ",
    "BEGIN FUNCTION",
    "Your response should use the following format",
    "Your output must strictly follow this format",
]

LEGACY_STD_TOOL_NAMES = {
    "bash",
    "edit_file",
    "execute_bash",
    "execute_ipython_cell",
    "generic_tool",
    "str_replace_editor",
    "stop",
    "submit",
}

DATASET_BRANCH_NAMES = {path.name for path in DATASET_PATH.iterdir() if path.is_dir()}


def sample_sft_files():
    return sorted(DATASET_PATH.glob("*/sample_sft/*.json"))


def sample_std_files():
    return sorted(DATASET_PATH.glob("*/sample_std.json"))


def message_text(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return json.dumps(content, ensure_ascii=False)


def non_system_messages(record):
    for message in record.get("messages", []):
        if message.get("role") != "system":
            yield message_text(message)
    for message in record.get("conversations", []):
        yield message_text(message)


@pytest.mark.parametrize("sample_path", sample_sft_files())
def test_sft_non_system_messages_do_not_repeat_agent_prompting(sample_path):
    records = json.loads(sample_path.read_text())
    violations = []
    for record_index, record in enumerate(records):
        for message_index, text in enumerate(non_system_messages(record)):
            for pattern in NON_SYSTEM_PROMPT_PATTERNS:
                if pattern in text:
                    violations.append((record_index, message_index, pattern))
    assert not violations, f"{sample_path} leaks generated prompt text outside system: {violations}"


@pytest.mark.parametrize("sample_path", sample_std_files())
def test_standardized_samples_use_agent_neutral_tool_names(sample_path):
    rows = json.loads(sample_path.read_text())
    violations = []
    for row_index, row in enumerate(rows):
        for step in row.get("steps", []):
            for tool_call in step.get("tool_calls") or []:
                function_name = tool_call.get("function_name")
                if function_name in LEGACY_STD_TOOL_NAMES:
                    violations.append((row_index, step.get("step_id"), function_name))
    assert not violations, f"{sample_path} contains legacy tool names: {violations}"


@pytest.mark.parametrize(
    "converter_path",
    sorted(AGENTS_PATH.glob("*/std_to_sft.py")) + sorted(AGENTS_PATH.glob("*/std_to_sft_mcp.py")),
)
def test_agent_sft_converters_do_not_branch_on_dataset_names(converter_path):
    tree = ast.parse(converter_path.read_text())
    constants = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)}
    leaked_dataset_names = sorted(
        DATASET_BRANCH_NAMES & {value for value in constants if isinstance(value, str)}
    )
    assert not leaked_dataset_names, (
        f"{converter_path} contains dataset-specific string constants: {leaked_dataset_names}"
    )
