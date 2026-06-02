#!/usr/bin/env python3
"""Adapt OpenHands SDK OpenAI SFT JSONL for LLaMA-Factory.

The OpenHands SDK SFT exporters write canonical OpenAI chat-completions tool
calls, where ``tool_calls[].function.arguments`` is a JSON string. That is the
right wire format, but LLaMA-Factory's Qwen 3.5 tool formatter expects parsed
argument objects after its OpenAI converter has read each function-call message.

This utility keeps the source SFT format canonical and performs a one-way
training adapter conversion:

* assistant messages with ``tool_calls`` become ``role="function_call"``
  messages whose ``content`` is a JSON string containing a list of functions;
* every function in that JSON has parsed object ``arguments``;
* nonessential OpenAI fields such as ``tool_call_id`` are dropped from messages
  to keep the Hugging Face Arrow schema stable;
* the top-level ``tools`` field is stringified so heterogeneous tool schemas do
  not become nested Arrow columns.

The output is intended for LLaMA-Factory with ``formatting: openai`` and tags
matching the defaults emitted by ``write_dataset_info``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_DATASET_NAME = "openhands_sdk_llamafactory"


def text_content(content: Any) -> str:
    """Convert OpenAI message content into the string LLaMA-Factory expects."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def parse_arguments(
    arguments: Any, *, record_id: str, message_index: int, call_index: int
) -> dict[str, Any]:
    """Return a dict of tool-call arguments from canonical OpenAI arguments."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON function.arguments in {record_id} "
                f"message {message_index} tool call {call_index}"
            ) from exc
    if not isinstance(arguments, dict):
        raise ValueError(
            f"function.arguments must decode to an object in {record_id} "
            f"message {message_index} tool call {call_index}: {type(arguments).__name__}"
        )
    return arguments


def adapt_tool_calls(
    record_id: str, message: dict[str, Any], message_index: int
) -> list[dict[str, Any]]:
    """Convert OpenAI tool calls to LLaMA-Factory function-call content."""
    functions: list[dict[str, Any]] = []
    for call_index, tool_call in enumerate(message.get("tool_calls") or []):
        function = tool_call.get("function")
        if not isinstance(function, dict):
            raise ValueError(
                f"tool_calls[{call_index}].function must be an object in "
                f"{record_id} message {message_index}"
            )
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"tool_calls[{call_index}].function.name must be a non-empty string "
                f"in {record_id} message {message_index}"
            )
        functions.append(
            {
                "name": name,
                "arguments": parse_arguments(
                    function.get("arguments", "{}"),
                    record_id=record_id,
                    message_index=message_index,
                    call_index=call_index,
                ),
            }
        )
    return functions


def adapt_message(record_id: str, message: dict[str, Any], message_index: int) -> dict[str, str]:
    """Adapt one OpenAI message to a schema-stable LLaMA-Factory message."""
    role = message.get("role")
    if not isinstance(role, str):
        raise ValueError(f"message role must be a string in {record_id} message {message_index}")
    if message.get("tool_calls"):
        functions = adapt_tool_calls(record_id, message, message_index)
        return {
            "role": "function_call",
            "content": json.dumps(functions, ensure_ascii=False),
        }
    return {
        "role": role,
        "content": text_content(message.get("content")),
    }


def adapt_record(record: dict[str, Any]) -> dict[str, Any]:
    """Adapt a single OpenHands SDK OpenAI SFT record for LLaMA-Factory."""
    record_id = str(record.get("id", "<unknown>"))
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"record {record_id} is missing a messages list")

    adapted: dict[str, Any] = {
        "id": record.get("id"),
        "messages": [
            adapt_message(record_id, message, index)
            for index, message in enumerate(messages)
            if isinstance(message, dict)
        ],
    }
    if len(adapted["messages"]) != len(messages):
        raise ValueError(f"record {record_id} contains a non-object message")

    tools = record.get("tools", "")
    if isinstance(tools, str):
        adapted["tools"] = tools
    elif tools is None:
        adapted["tools"] = ""
    else:
        adapted["tools"] = json.dumps(tools, ensure_ascii=False)

    if "metadata" in record:
        adapted["metadata"] = record["metadata"]
    return adapted


def convert_jsonl(input_path: Path, output_path: Path) -> int:
    """Convert an OpenHands SDK OpenAI SFT JSONL file."""
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        input_path.open(encoding="utf-8") as in_handle,
        output_path.open("w", encoding="utf-8") as out_handle,
    ):
        for line_number, line in enumerate(in_handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                adapted = adapt_record(record)
            except Exception as exc:
                raise ValueError(f"Failed to adapt {input_path}:{line_number}") from exc
            out_handle.write(json.dumps(adapted, ensure_ascii=False) + "\n")
            count += 1
    return count


def dataset_info(dataset_name: str, file_name: str) -> dict[str, Any]:
    """Return LLaMA-Factory dataset_info.json content for adapted records."""
    return {
        dataset_name: {
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
            "file_name": file_name,
        }
    }


def write_dataset_info(path: Path, *, dataset_name: str, file_name: str) -> None:
    """Write a LLaMA-Factory dataset_info.json for the adapted JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset_info(dataset_name, file_name), indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="OpenHands SDK OpenAI SFT JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Adapted LLaMA-Factory JSONL")
    parser.add_argument(
        "--dataset-info",
        type=Path,
        help="Optional dataset_info.json path to write or update for the adapted file.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Dataset key to write when --dataset-info is provided.",
    )
    args = parser.parse_args()

    count = convert_jsonl(args.input, args.output)
    if args.dataset_info:
        write_dataset_info(
            args.dataset_info,
            dataset_name=args.dataset_name,
            file_name=args.output.name,
        )
    print(json.dumps({"input": str(args.input), "output": str(args.output), "records": count}))


if __name__ == "__main__":
    main()
