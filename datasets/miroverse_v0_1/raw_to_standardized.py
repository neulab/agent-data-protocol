import json
import re
import sys
from typing import Any

from schema_raw import SchemaRaw

from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

MCP_CALL_RE = re.compile(r"<use_mcp_tool>\s*(.*?)\s*</use_mcp_tool>", re.DOTALL)
TAG_RE_TEMPLATE = r"<{tag}>\s*(.*?)\s*</{tag}>"


def _extract_tag(block: str, tag: str) -> str | None:
    match = re.search(TAG_RE_TEMPLATE.format(tag=re.escape(tag)), block, re.DOTALL)
    return match.group(1).strip() if match else None


def _parse_arguments(raw_arguments: str | None) -> dict[str, Any] | str:
    if raw_arguments is None:
        return {}
    text = raw_arguments.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        print(
            f"Warning: Failed to parse MCP arguments as JSON: {text[:100]}... Error: {exc}",
            file=sys.stderr,
        )
        return text


def _convert_assistant_message(content: str):
    """Convert assistant text with optional MCP XML tags into ADP actions."""
    content = content.strip()
    if not content:
        return []

    match = MCP_CALL_RE.search(content)
    if not match:
        return [MessageAction(content=content, description=None)]

    before = content[: match.start()].strip()
    after = content[match.end() :].strip()
    block = match.group(1)
    server_name = _extract_tag(block, "server_name") or ""
    tool_name = _extract_tag(block, "tool_name") or ""
    arguments = _parse_arguments(_extract_tag(block, "arguments"))

    converted = [
        ApiAction(
            function="use_mcp_tool",
            kwargs={
                "server_name": repr(server_name),
                "tool_name": repr(tool_name),
                "arguments": repr(arguments),
            },
            description=before or None,
        )
    ]
    if after:
        converted.append(MessageAction(content=after, description=None))
    return converted


def _convert_message(message, previous_was_tool_call: bool):
    role = message.role
    content = message.content
    if role == "system":
        return []
    if role == "assistant":
        return _convert_assistant_message(content)
    if previous_was_tool_call:
        return [TextObservation(content=content, source="environment")]
    return [TextObservation(content=content, source="user")]


def _mark_final_answer(content):
    """Wrap the final assistant message as an ADP finish action."""
    if not content:
        return
    last = content[-1]
    if isinstance(last, MessageAction) and "<finish>" not in last.content:
        last.content = f"<finish> {last.content} </finish>"


for line in sys.stdin:
    raw_data = json.loads(line)
    data = SchemaRaw(**raw_data)
    content = []
    previous_was_tool_call = False
    system_prompt = "\n\n".join(
        message.content for message in data.messages if message.role == "system"
    )

    for message in data.messages:
        converted = _convert_message(message, previous_was_tool_call)
        content.extend(converted)
        previous_was_tool_call = bool(converted) and isinstance(converted[-1], ApiAction)

    _mark_final_answer(content)

    details = {"split": data.split or ""}
    if system_prompt:
        details["system_prompt"] = system_prompt

    trajectory = Trajectory(
        id=data.id or f"{data.split or 'miroverse'}-unknown",
        content=content,
        details=details,
    )
    print(json.dumps(trajectory.model_dump(), ensure_ascii=False))
