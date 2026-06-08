import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from schema.dataset_metadata import (
    DatasetMetadata,
    OpenAIFunctionSpec,
    OpenAIToolSpec,
    custom_tool_names,
    load_dataset_metadata,
)

ROOT = Path(__file__).parent.parent
DATASET_PATH = ROOT / "datasets"
SDK_SAMPLE_DATASETS = [
    "agenttuning_alfworld",
    "agenttuning_kg",
    "agenttuning_mind2web",
    "agenttuning_os",
    "agenttuning_webshop",
]


def content_text(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "\n".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def tool_call_names(record):
    return [
        tool_call["function"]["name"]
        for message in record["messages"]
        if message.get("role") == "assistant"
        for tool_call in message.get("tool_calls", [])
    ]


def trajectory_id(row: dict) -> str:
    return str(row.get("trajectory_id") or row["id"])


def assert_sdk_chat_record(record):
    assert record["messages"][0]["role"] == "system"
    assert content_text(record["messages"][0]).startswith("You are OpenHands agent")
    assert record["metadata"]["generation"] == "openhands_sdk_events"
    tool_names = {tool["function"]["name"] for tool in record["tools"]}
    pending_tool_call_ids = []
    for message in record["messages"]:
        assert message["role"] in {"system", "user", "assistant", "tool"}
        if message["role"] == "assistant":
            for tool_call in message.get("tool_calls", []):
                assert tool_call["type"] == "function"
                assert tool_call["function"]["name"] in tool_names
                json.loads(tool_call["function"]["arguments"])
                pending_tool_call_ids.append(tool_call["id"])
        if message["role"] == "tool":
            assert message["tool_call_id"] in pending_tool_call_ids
            pending_tool_call_ids.remove(message["tool_call_id"])
    assert not pending_tool_call_ids


def run_converter(dataset: str, rows: list[dict], args: list[str] | None = None, check=True):
    env = os.environ.copy()
    env["OPENHANDS_SUPPRESS_BANNER"] = "1"
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    env["MY_DATASET"] = dataset
    proc = subprocess.run(
        [sys.executable, str(ROOT / "agents/openhands_sdk/std_to_sft.py"), *(args or [])],
        input="\n".join(json.dumps(row) for row in rows),
        text=True,
        capture_output=True,
        check=check,
        env=env,
    )
    return [json.loads(line) for line in proc.stdout.splitlines() if line]


def patch_condensation_llm(monkeypatch, summary="[ADP condensation test summary]"):
    from litellm.types.utils import Choices, ModelResponse
    from litellm.types.utils import Message as LiteLLMMessage
    from openhands.sdk import Message, TextContent
    from openhands.sdk.llm.llm_response import LLMResponse
    from openhands.sdk.llm.utils.metrics import MetricsSnapshot, TokenUsage

    from agents.openhands_sdk import condensation_sft

    def fake_completion(
        self,
        messages,
        tools=None,
        _return_metrics=False,
        add_security_risk_prediction=False,
        on_token=None,
        **kwargs,
    ):
        self._captured_messages.append(messages)
        return LLMResponse(
            message=Message(role="assistant", content=[TextContent(text=summary)]),
            metrics=MetricsSnapshot(
                model_name=self.model,
                accumulated_cost=0.0,
                max_budget_per_task=None,
                accumulated_token_usage=TokenUsage(
                    model=self.model,
                    prompt_tokens=0,
                    completion_tokens=0,
                ),
            ),
            raw_response=ModelResponse(
                id="adp-condensation-test-response",
                choices=[
                    Choices(
                        message=LiteLLMMessage(role="assistant", content=summary),
                        index=0,
                        finish_reason="stop",
                    )
                ],
                created=0,
                model="test-model",
                object="chat.completion",
            ),
        )

    monkeypatch.setattr(condensation_sft.PromptCapturingLLM, "completion", fake_completion)


def test_openhands_sdk_generated_samples_are_sdk_chat_records():
    for dataset in SDK_SAMPLE_DATASETS:
        records = json.loads(
            (DATASET_PATH / dataset / "sample_sft" / "openhands_sdk.json").read_text()
        )
        std_rows = json.loads((DATASET_PATH / dataset / "sample_std.json").read_text())
        metadata = load_dataset_metadata(dataset, required=True)
        assert [record["id"] for record in records] == [trajectory_id(row) for row in std_rows]
        assert metadata.custom_tools or metadata.code_enabled or metadata.browser_enabled
        for record in records:
            assert_sdk_chat_record(record)


def test_openhands_sdk_converter_uses_metadata_custom_tools():
    record = json.loads(
        (DATASET_PATH / "agenttuning_alfworld" / "sample_sft" / "openhands_sdk.json").read_text()
    )[0]
    metadata = load_dataset_metadata("agenttuning_alfworld", required=True)
    declared_tools = {tool["function"]["name"] for tool in record["tools"]}
    assert custom_tool_names(metadata) <= declared_tools
    assert {"go", "take", "put"} <= set(tool_call_names(record))


def test_openhands_sdk_converter_preserves_webshop_label_clicks():
    records = json.loads(
        (DATASET_PATH / "agenttuning_webshop" / "sample_sft" / "openhands_sdk.json").read_text()
    )
    click_arguments = [
        json.loads(tool_call["function"]["arguments"])
        for record in records
        for message in record["messages"]
        if message.get("role") == "assistant"
        for tool_call in message.get("tool_calls", [])
        if tool_call["function"]["name"] == "click"
    ]

    assert {"element": "Buy Now"} in click_arguments
    assert all("index" not in arguments for arguments in click_arguments)


def test_openhands_sdk_converter_regenerates_first_record():
    dataset = "agenttuning_os"
    source = json.loads((DATASET_PATH / dataset / "sample_std.json").read_text())[:1]
    generated = run_converter(dataset, source)
    fixture = json.loads(
        (DATASET_PATH / dataset / "sample_sft" / "openhands_sdk.json").read_text()
    )[:1]
    assert generated == fixture
    assert tool_call_names(generated[0])[:2] == ["terminal", "terminal"]


def test_openhands_sdk_converter_reads_dataset_at_call_time(monkeypatch):
    from agents.openhands_sdk import std_to_sft

    dataset = "agenttuning_alfworld"
    source = json.loads((DATASET_PATH / dataset / "sample_std.json").read_text())[0]
    monkeypatch.setenv("MY_DATASET", dataset)

    generated = std_to_sft.process_row(json.dumps(source), "gpt-4o-mini")

    assert generated["metadata"]["source_dataset"] == dataset


def test_openhands_sdk_converter_rejects_unsupported_legacy_cli_flags():
    dataset = "agenttuning_alfworld"
    source = json.loads((DATASET_PATH / dataset / "sample_std.json").read_text())[:1]

    env = os.environ.copy()
    env["OPENHANDS_SUPPRESS_BANNER"] = "1"
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    env["MY_DATASET"] = dataset
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "agents/openhands_sdk/std_to_sft.py"),
            "--is_web",
            "no",
        ],
        input="\n".join(json.dumps(row) for row in source),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert proc.returncode != 0
    assert "--is_web and --api_env are not supported" in proc.stderr


def test_openhands_sdk_converter_rejects_conflicting_custom_tool_schemas(monkeypatch):
    from agents.openhands_sdk import std_to_sft

    monkeypatch.setattr(std_to_sft, "_REGISTERED_METADATA_TOOL_SPECS", {})
    registered_tools = {}
    monkeypatch.setattr(
        std_to_sft,
        "register_tool",
        lambda name, tool_definition: registered_tools.setdefault(name, tool_definition),
    )

    tool_name = "adp_conflict_test_tool"
    first_metadata = DatasetMetadata(
        custom_tools=[
            OpenAIToolSpec(
                function=OpenAIFunctionSpec(
                    name=tool_name,
                    parameters={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                )
            )
        ]
    )
    second_metadata = DatasetMetadata(
        custom_tools=[
            OpenAIToolSpec(
                function=OpenAIFunctionSpec(
                    name=tool_name,
                    parameters={
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"],
                    },
                )
            )
        ]
    )

    std_to_sft.register_metadata_tools(first_metadata)
    assert tool_name in registered_tools
    with pytest.raises(ValueError, match="different schema"):
        std_to_sft.register_metadata_tools(second_metadata)


def test_openhands_sdk_condensation_utility_emits_llm_summaries_after_trajectories(
    monkeypatch,
):
    from agents.openhands_sdk import condensation_sft

    patch_condensation_llm(monkeypatch)
    dataset = "agenttuning_os"
    source = json.loads((DATASET_PATH / dataset / "sample_std.json").read_text())[0]

    records = condensation_sft.process_row(
        json.dumps(source),
        max_tokens=2000,
        model="gpt-4o-mini",
        dataset_name=dataset,
    )

    trajectory_records = [
        record for record in records if record["metadata"].get("record_type") == "trajectory"
    ]
    prompt_records = [
        record
        for record in records
        if record["metadata"]["generation"] == "openhands_sdk_condensation_prompt"
    ]

    assert prompt_records
    assert len(trajectory_records) == len(prompt_records) + 1
    assert records[:3] == [
        trajectory_records[0],
        prompt_records[0],
        trajectory_records[1],
    ]
    source_id = trajectory_id(source)
    assert trajectory_records[0]["id"] == f"{source_id}__trajectory_0001"
    assert prompt_records[0]["id"] == f"{source_id}__condensation_0001"
    assert trajectory_records[1]["id"] == f"{source_id}__trajectory_0002"
    assert prompt_records[0]["messages"] == [
        {
            "role": "user",
            "content": prompt_records[0]["messages"][0]["content"],
        },
        {"role": "assistant", "content": "[ADP condensation test summary]"},
    ]
    assert prompt_records[0]["messages"][0]["content"].startswith(
        "You are maintaining a context-aware state summary"
    )
    assert "<EVENT>" in prompt_records[0]["messages"][0]["content"]
    assert prompt_records[0]["metadata"]["prompt_token_count_before_condensation"] > 2000
    assert prompt_records[0]["metadata"]["forgotten_event_count"] > 0
    assert prompt_records[0]["metadata"]["condensation_output"] == "llm"


def test_openhands_sdk_condensation_utility_rejects_non_llm_output_modes():
    dataset = "agenttuning_os"
    source = json.loads((DATASET_PATH / dataset / "sample_std.json").read_text())[:1]

    env = os.environ.copy()
    env["OPENHANDS_SUPPRESS_BANNER"] = "1"
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    env["MY_DATASET"] = dataset
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "agents/openhands_sdk/condensation_sft.py"),
            "--max-tokens",
            "2000",
            "--condensation-output",
            "placeholder",
        ],
        input="\n".join(json.dumps(row) for row in source),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert proc.returncode != 0
    assert "unrecognized arguments: --condensation-output" in proc.stderr


def test_openhands_sdk_condensation_utility_identity_maps_short_trajectories():
    from agents.openhands_sdk import condensation_sft

    dataset = "agenttuning_os"
    source = json.loads((DATASET_PATH / dataset / "sample_std.json").read_text())[0]

    records = condensation_sft.process_row(
        json.dumps(source),
        max_tokens=100_000,
        model="gpt-4o-mini",
        dataset_name=dataset,
        include_trajectories=False,
    )

    assert len(records) == 1
    assert records[0]["id"] == f"{trajectory_id(source)}__trajectory_0001"
    assert records[0]["metadata"]["generation"] == "openhands_sdk_events"
    assert records[0]["metadata"]["record_type"] == "trajectory"


def test_openhands_sdk_condensation_loader_backfills_legacy_tool_call_links():
    from agents.openhands_sdk import condensation_sft
    from schema.trajectory import Trajectory

    row = {
        "id": "legacy_missing_tool_call_ids",
        "content": [
            {"class_": "text_observation", "content": "do something", "source": "user"},
            {
                "class_": "code_action",
                "language": "bash",
                "content": "pwd",
                "description": "check current directory",
            },
            {"class_": "text_observation", "content": "/workspace", "source": "user"},
        ],
    }

    with pytest.raises(ValueError, match="tool_call_id"):
        Trajectory(**row)

    trajectory = condensation_sft.load_trajectory(json.dumps(row))

    action = trajectory.content[1]
    observation = trajectory.content[2]
    assert action.tool_call_id == "call_000001"
    assert observation.tool_call_id == "call_000001"
    assert observation.source == "environment"


def test_openhands_sdk_converter_preserves_file_editor_string_arguments():
    from agents.openhands_sdk import std_to_sft
    from schema.action.api import ApiAction

    action = ApiAction(
        function="str_replace_editor",
        kwargs={
            "command": "str_replace",
            "path": "/workspace/file.py",
            "old_str": "24",
            "new_str": 25,
        },
    )

    tool_name, arguments = std_to_sft.map_api_action(action, DatasetMetadata())

    assert tool_name == "file_editor"
    assert arguments["old_str"] == "24"
    assert arguments["new_str"] == "25"


def test_openhands_sdk_converter_maps_python_code_action_to_terminal_heredoc():
    from agents.openhands_sdk import std_to_sft
    from schema.action.code import CodeAction

    action = CodeAction(language="python", content="print('hello')", description="run python")

    tool_name, arguments = std_to_sft.map_code_action(action)

    assert tool_name == "terminal"
    assert arguments["command"].startswith("python <<'ADP_PYTHON_")
    assert "\nprint('hello')\n" in arguments["command"]


def test_openhands_sdk_converter_rejects_mysql_code_action():
    from agents.openhands_sdk import std_to_sft
    from schema.action.code import CodeAction

    action = CodeAction(language="mysql", content="SELECT 1;", description="run mysql")

    with pytest.raises(ValueError, match="mysql"):
        std_to_sft.map_code_action(action)


def test_openhands_sdk_dataset_tool_observation_truncates_large_outputs():
    from openhands.tools.terminal.definition import MAX_CMD_OUTPUT_SIZE

    from agents.openhands_sdk.std_to_sft import DatasetToolObservation

    observation = DatasetToolObservation(output="A" * (MAX_CMD_OUTPUT_SIZE + 1000))
    [content] = observation.to_llm_content

    assert len(content.text) == MAX_CMD_OUTPUT_SIZE
    assert "response clipped" in content.text
