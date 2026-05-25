import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATASET_PATH = ROOT / "datasets"


def run_converter(dataset: str, sample_name: str = "sample_std.json"):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    env["MY_DATASET"] = dataset
    sample_path = DATASET_PATH / dataset / sample_name
    jsonl_proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/json_to_jsonl.py")],
        input=sample_path.read_text(),
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )
    converter_proc = subprocess.run(
        [sys.executable, str(ROOT / "agents/openhands_sdk/std_to_sft.py")],
        input=jsonl_proc.stdout,
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )
    return [json.loads(line) for line in converter_proc.stdout.splitlines() if line]


def run_converter_rows(rows: list[dict]):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    converter_proc = subprocess.run(
        [sys.executable, str(ROOT / "agents/openhands_sdk/std_to_sft.py")],
        input="\n".join(json.dumps(row) for row in rows),
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )
    return [json.loads(line) for line in converter_proc.stdout.splitlines() if line]


def get_dataset_dirs():
    return sorted(
        path.name
        for path in DATASET_PATH.iterdir()
        if path.is_dir() and (path / "sample_std.json").exists()
    )


def content_text(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "\n".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def assert_openai_chat_record(record):
    assert record["messages"][0]["role"] == "system"
    assert content_text(record["messages"][0]).startswith("You are OpenHands agent")
    assert "tools" in record
    assert "conversations" not in record

    pending_tool_call_ids = []
    tool_names = {tool["function"]["name"] for tool in record["tools"]}
    assert {"terminal", "file_editor", "task_tracker", "finish", "think"} <= tool_names

    for message in record["messages"]:
        assert message["role"] in {"system", "user", "assistant", "tool"}
        assert "from" not in message
        assert "value" not in message
        if message["role"] == "assistant" and "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                assert tool_call["type"] == "function"
                assert tool_call["function"]["name"] in tool_names
                json.loads(tool_call["function"]["arguments"])
                pending_tool_call_ids.append(tool_call["id"])
        if message["role"] == "tool":
            assert message["tool_call_id"] in pending_tool_call_ids
            pending_tool_call_ids.remove(message["tool_call_id"])

    assert not pending_tool_call_ids


def test_openhands_sdk_converter_uses_native_tool_calls():
    records = run_converter("swe-smith")
    record = records[0]

    assert_openai_chat_record(record)
    assistant_tool_message = next(
        message
        for message in record["messages"]
        if message["role"] == "assistant" and "tool_calls" in message
    )
    tool_call = assistant_tool_message["tool_calls"][0]
    assert tool_call["function"]["name"] == "terminal"
    assert "<function=" not in json.dumps(record["messages"])
    args = json.loads(tool_call["function"]["arguments"])
    assert args["security_risk"] == "LOW"


def test_openhands_sdk_converter_converts_finish_to_tool_call():
    records = run_converter("agenttuning_mind2web")
    record = records[0]

    assert_openai_chat_record(record)
    assistant_tool_message = record["messages"][-2]
    tool_result_message = record["messages"][-1]
    tool_call = assistant_tool_message["tool_calls"][0]
    assert tool_call["function"]["name"] == "finish"
    assert "security_risk" not in json.loads(tool_call["function"]["arguments"])
    assert tool_result_message["role"] == "tool"
    assert tool_result_message["name"] == "finish"


def test_openhands_sdk_converter_parses_legacy_finish_xml():
    records = run_converter("swe-play-trajectories")
    record = records[0]

    assert_openai_chat_record(record)
    assert not any(
        message.get("role") == "assistant" and "<function=" in content_text(message)
        for message in record["messages"]
    )
    finish_call = record["messages"][-2]["tool_calls"][0]
    assert finish_call["function"]["name"] == "finish"


def test_openhands_sdk_converter_maps_followup_observation_to_tool_result():
    records = run_converter("agenttuning_alfworld")
    record = records[0]

    assert_openai_chat_record(record)
    go_call_message = next(
        message
        for message in record["messages"]
        if message.get("tool_calls") and message["tool_calls"][0]["function"]["name"] == "go"
    )
    go_call_id = go_call_message["tool_calls"][0]["id"]
    go_result_index = next(
        index
        for index, message in enumerate(record["messages"])
        if message.get("role") == "tool" and message.get("tool_call_id") == go_call_id
    )
    go_result = record["messages"][go_result_index]
    assert "On the shelf 1" in content_text(go_result)
    assert record["messages"][go_result_index + 1]["role"] == "assistant"


def test_openhands_sdk_converter_preserves_eto_webshop_tool_calls():
    records = run_converter("eto")
    record = records[0]

    tool_names = [
        tool_call["function"]["name"]
        for message in record["messages"]
        if message.get("role") == "assistant"
        for tool_call in message.get("tool_calls", [])
    ]

    assert tool_names == ["search", "browser_click", "browser_click", "finish"]


def test_openhands_sdk_converter_preserves_agenttuning_mind2web_tool_calls():
    records = run_converter("agenttuning_mind2web")
    tool_names = [
        [
            tool_call["function"]["name"]
            for message in record["messages"]
            if message.get("role") == "assistant"
            for tool_call in message.get("tool_calls", [])
        ]
        for record in records
    ]

    assert tool_names == [
        ["browser_click", "finish"],
        ["browser_type", "finish"],
        ["browser_type", "finish"],
        ["browser_click", "finish"],
        ["browser_type", "finish"],
    ]


def test_openhands_sdk_converter_preserves_explicit_tool_call_ids():
    records = run_converter_rows(
        [
            {
                "id": "explicit-tool-call-ids",
                "content": [
                    {
                        "class_": "text_observation",
                        "content": "Search for ADP and inspect the repo.",
                        "source": "user",
                    },
                    {
                        "class_": "api_action",
                        "tool_call_id": "source_call_search",
                        "function": "search",
                        "kwargs": {"query": "agent data protocol"},
                    },
                    {
                        "class_": "code_action",
                        "tool_call_id": "source_call_shell",
                        "language": "bash",
                        "content": "ls",
                        "description": None,
                    },
                    {
                        "class_": "text_observation",
                        "tool_call_id": "source_call_shell",
                        "content": "README.md\nschema",
                        "source": "environment",
                    },
                    {
                        "class_": "text_observation",
                        "tool_call_id": "source_call_search",
                        "content": "ADP standardizes agent trajectories.",
                        "source": "environment",
                    },
                ],
            }
        ]
    )
    record = records[0]

    assert_openai_chat_record(record)
    tool_call_ids = [
        tool_call["id"]
        for message in record["messages"]
        for tool_call in message.get("tool_calls", [])
    ]
    tool_result_ids = [
        message["tool_call_id"] for message in record["messages"] if message["role"] == "tool"
    ]
    assert tool_call_ids == ["source_call_search", "source_call_shell"]
    assert tool_result_ids == ["source_call_shell", "source_call_search"]


def test_openhands_sdk_samples_align_with_standardized_ids_when_present():
    for dataset in get_dataset_dirs():
        sample_sft_path = DATASET_PATH / dataset / "sample_sft" / "openhands_sdk.json"
        if not sample_sft_path.exists():
            continue
        std_data = json.loads((DATASET_PATH / dataset / "sample_std.json").read_text())
        sft_data = json.loads(sample_sft_path.read_text())
        assert [row["id"] for row in sft_data] == [row["id"] for row in std_data]
        for record in sft_data:
            assert_openai_chat_record(record)
