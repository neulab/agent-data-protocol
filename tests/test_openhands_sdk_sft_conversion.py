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


def test_openhands_sdk_generated_samples_are_sdk_chat_records():
    for dataset in SDK_SAMPLE_DATASETS:
        records = json.loads(
            (DATASET_PATH / dataset / "sample_sft" / "openhands_sdk.json").read_text()
        )
        std_rows = json.loads((DATASET_PATH / dataset / "sample_std.json").read_text())
        metadata = load_dataset_metadata(dataset, required=True)
        assert [record["id"] for record in records] == [row["id"] for row in std_rows]
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

    dataset = "agenttuning_os"
    source = json.loads((DATASET_PATH / dataset / "sample_std.json").read_text())[0]
    monkeypatch.setenv("MY_DATASET", dataset)

    generated = std_to_sft.process_row(json.dumps(source), "gpt-4o-mini")

    assert generated["metadata"]["source_dataset"] == dataset


def test_openhands_sdk_converter_rejects_unsupported_legacy_cli_flags():
    dataset = "agenttuning_os"
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


def test_openhands_sdk_condensation_utility_emits_prompts_after_trajectories():
    from agents.openhands_sdk import condensation_sft

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
    assert trajectory_records[0]["id"] == f"{source['id']}__trajectory_0001"
    assert prompt_records[0]["id"] == f"{source['id']}__condensation_0001"
    assert trajectory_records[1]["id"] == f"{source['id']}__trajectory_0002"
    assert prompt_records[0]["messages"] == [
        {
            "role": "user",
            "content": prompt_records[0]["messages"][0]["content"],
        }
    ]
    assert prompt_records[0]["messages"][0]["content"].startswith(
        "You are maintaining a context-aware state summary"
    )
    assert "<EVENT>" in prompt_records[0]["messages"][0]["content"]
    assert prompt_records[0]["metadata"]["prompt_token_count_before_condensation"] > 2000
    assert prompt_records[0]["metadata"]["forgotten_event_count"] > 0


def test_openhands_sdk_condensation_utility_can_include_placeholder_output():
    from agents.openhands_sdk import condensation_sft

    dataset = "agenttuning_os"
    source = json.loads((DATASET_PATH / dataset / "sample_std.json").read_text())[0]

    records = condensation_sft.process_row(
        json.dumps(source),
        max_tokens=2000,
        model="gpt-4o-mini",
        dataset_name=dataset,
        include_trajectories=False,
        condensation_output="placeholder",
    )

    assert records
    assert records[0]["messages"][-1] == {
        "role": "assistant",
        "content": "[ADP condensation placeholder #1]",
    }
    assert records[0]["metadata"]["condensation_output"] == "placeholder"


def test_openhands_sdk_condensation_utility_skips_short_trajectories():
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

    assert records == []
