import json
import subprocess
import sys

import pytest

from agents.openhands_sdk.sft_to_llamafactory import adapt_record, dataset_info


def test_adapt_record_converts_openai_tool_calls_to_function_messages():
    record = {
        "id": "example-1",
        "messages": [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "List files"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": json.dumps({"command": "ls", "timeout": 30}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "terminal",
                "content": "README.md",
            },
            {"role": "assistant", "content": "Done"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "metadata": {"source_dataset": "unit"},
    }

    adapted = adapt_record(record)

    assert adapted["messages"] == [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "List files"},
        {
            "role": "function_call",
            "content": json.dumps(
                [{"name": "terminal", "arguments": {"command": "ls", "timeout": 30}}],
                ensure_ascii=False,
            ),
        },
        {"role": "tool", "content": "README.md"},
        {"role": "assistant", "content": "Done"},
    ]
    assert isinstance(adapted["tools"], str)
    assert json.loads(adapted["tools"])[0]["function"]["name"] == "terminal"
    assert adapted["metadata"] == {"source_dataset": "unit"}



def test_adapt_record_rejects_non_object_arguments():
    record = {
        "id": "bad-args",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "terminal",
                            "arguments": json.dumps(["not", "an", "object"]),
                        }
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValueError, match="must decode to an object"):
        adapt_record(record)


def test_dataset_info_uses_llamafactory_openai_tags():
    info = dataset_info("demo", "demo.jsonl")

    assert info == {
        "demo": {
            "formatting": "openai",
            "columns": {"messages": "messages", "tools": "tools"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "observation_tag": "tool",
                "function_tag": "function_call",
                "system_tag": "system",
            },
            "file_name": "demo.jsonl",
        }
    }


def test_cli_converts_jsonl_and_writes_dataset_info(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    info_path = tmp_path / "dataset_info.json"
    input_record = {
        "id": "cli-1",
        "messages": [
            {"role": "user", "content": "Run command"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "terminal",
                            "arguments": json.dumps({"command": "pwd"}),
                        }
                    }
                ],
            },
        ],
        "tools": [],
    }
    input_path.write_text(json.dumps(input_record) + "\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agents.openhands_sdk.sft_to_llamafactory",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--dataset-info",
            str(info_path),
            "--dataset-name",
            "cli_demo",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert json.loads(result.stdout)["records"] == 1
    [adapted] = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert adapted["messages"][1]["role"] == "function_call"
    assert json.loads(adapted["messages"][1]["content"])[0] == {
        "name": "terminal",
        "arguments": {"command": "pwd"},
    }
    assert json.loads(info_path.read_text())["cli_demo"]["file_name"] == "output.jsonl"
