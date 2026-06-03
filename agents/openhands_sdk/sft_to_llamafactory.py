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
  not become nested Arrow columns;
* adjacent prompt-side messages (``user`` and ``tool``), which are valid
  OpenAI chat history but not accepted by LLaMA-Factory's paired-turn converter,
  are merged;
* when requested, OpenAI-valid conversation prefixes are converted into
  trainable prefixes by trimming trailing prompt-side messages (for example a
  final tool response) so the adapted record ends on an assistant/function turn;
* literal media tags are escaped so Qwen-VL/LLaMA-Factory does not treat source
  text such as XML ``<image>`` blocks as multimodal placeholders.

The output is intended for LLaMA-Factory with ``formatting: openai`` and tags
matching the defaults emitted by ``write_dataset_info``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_DATASET_NAME = "openhands_sdk_llamafactory"
PROMPT_ROLES = {"user", "tool"}
RESPONSE_ROLES = {"assistant", "function_call"}
MEDIA_TAG_REPLACEMENTS = {
    "<image>": "&lt;image&gt;",
    "</image>": "&lt;/image&gt;",
    "<video>": "&lt;video&gt;",
    "</video>": "&lt;/video&gt;",
    "<audio>": "&lt;audio&gt;",
    "</audio>": "&lt;/audio&gt;",
}


class UntrainableRecordError(ValueError):
    """Raised when a record has no assistant/function response to train on."""


def escape_media_tags(text: str) -> str:
    """Escape literal media tags reserved by multimodal chat templates."""
    for old, new in MEDIA_TAG_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def escape_media_tags_in_json(value: Any) -> Any:
    """Recursively escape reserved media tags inside JSON-compatible values."""
    if isinstance(value, str):
        return escape_media_tags(value)
    if isinstance(value, list):
        return [escape_media_tags_in_json(item) for item in value]
    if isinstance(value, dict):
        return {key: escape_media_tags_in_json(item) for key, item in value.items()}
    return value


def text_content(content: Any) -> str:
    """Convert OpenAI message content into the string LLaMA-Factory expects."""
    if content is None:
        return ""
    if isinstance(content, str):
        return escape_media_tags(content)
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
        return escape_media_tags("\n".join(parts))
    return escape_media_tags(str(content))


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
    return escape_media_tags_in_json(arguments)


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


def is_prompt_side(message: dict[str, str]) -> bool:
    return message["role"] in PROMPT_ROLES


def merge_adjacent_prompt_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Merge adjacent user/tool messages for LLaMA-Factory paired-turn conversion."""
    merged: list[dict[str, str]] = []
    for message in messages:
        if merged and is_prompt_side(message) and is_prompt_side(merged[-1]):
            merged[-1]["content"] = f"{merged[-1]['content']}\n\n{message['content']}"
        else:
            merged.append(dict(message))
    return merged


def trainable_prefix(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Trim a LLaMA-Factory record to the latest trainable assistant/function turn.

    Many OpenAI Chat Completions histories are valid prefixes ending in ``tool``
    or ``user``. They are useful context, but the single-record SFT format needs
    the final message to be a response-side message that can become the label.
    """
    if not messages:
        raise UntrainableRecordError("record contains no messages")

    prefix = [dict(messages[0])] if messages[0]["role"] == "system" else []
    body = messages[1:] if prefix else messages
    last_response_index: int | None = None
    for index, message in enumerate(body):
        if message["role"] in RESPONSE_ROLES:
            last_response_index = index

    if last_response_index is None:
        raise UntrainableRecordError("record contains no trainable assistant/function response")

    return prefix + [dict(message) for message in body[: last_response_index + 1]]


def adapt_record(record: dict[str, Any], *, trim_to_trainable: bool = False) -> dict[str, Any]:
    """Adapt a single OpenHands SDK OpenAI SFT record for LLaMA-Factory."""
    record_id = str(record.get("id", "<unknown>"))
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"record {record_id} is missing a messages list")

    adapted_messages = [
        adapt_message(record_id, message, index)
        for index, message in enumerate(messages)
        if isinstance(message, dict)
    ]
    if len(adapted_messages) != len(messages):
        raise ValueError(f"record {record_id} contains a non-object message")

    merged_messages = merge_adjacent_prompt_messages(adapted_messages)
    adapted: dict[str, Any] = {
        "id": record.get("id"),
        "messages": trainable_prefix(merged_messages) if trim_to_trainable else merged_messages,
    }

    tools = record.get("tools", "")
    if isinstance(tools, str):
        adapted["tools"] = escape_media_tags(tools)
    elif tools is None:
        adapted["tools"] = ""
    else:
        adapted["tools"] = escape_media_tags(json.dumps(tools, ensure_ascii=False))

    if "metadata" in record:
        adapted["metadata"] = record["metadata"]
    return adapted


def convert_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    trim_to_trainable: bool = False,
    skip_untrainable: bool = False,
) -> dict[str, int]:
    """Convert an OpenHands SDK OpenAI SFT JSONL file."""
    stats = {"read": 0, "written": 0, "skipped_untrainable": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        input_path.open(encoding="utf-8") as in_handle,
        output_path.open("w", encoding="utf-8") as out_handle,
    ):
        for line_number, line in enumerate(in_handle, 1):
            if not line.strip():
                continue
            stats["read"] += 1
            try:
                record = json.loads(line)
                adapted = adapt_record(record, trim_to_trainable=trim_to_trainable)
            except UntrainableRecordError:
                if skip_untrainable:
                    stats["skipped_untrainable"] += 1
                    continue
                raise
            except Exception as exc:
                raise ValueError(f"Failed to adapt {input_path}:{line_number}") from exc
            out_handle.write(json.dumps(adapted, ensure_ascii=False) + "\n")
            stats["written"] += 1
    return stats


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
    parser.add_argument(
        "--trim-to-trainable",
        action="store_true",
        help="Trim OpenAI conversation prefixes so each output record ends on an assistant/function turn.",
    )
    parser.add_argument(
        "--skip-untrainable",
        action="store_true",
        help="Skip records that still have no assistant/function turn after trimming.",
    )
    args = parser.parse_args()

    stats = convert_jsonl(
        args.input,
        args.output,
        trim_to_trainable=args.trim_to_trainable,
        skip_untrainable=args.skip_untrainable,
    )
    if args.dataset_info:
        write_dataset_info(
            args.dataset_info,
            dataset_name=args.dataset_name,
            file_name=args.output.name,
        )
    print(json.dumps({"input": str(args.input), "output": str(args.output), **stats}))


if __name__ == "__main__":
    main()
