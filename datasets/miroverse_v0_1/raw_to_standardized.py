import json
import re
import sys
from typing import Any

from extract_raw import (
    extract_available_tools_from_messages,
    sanitize_identifier,
    tool_function_name,
)
from schema_raw import SchemaRaw

from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.tool_call_links import create_trajectory_with_tool_call_links

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


def _tool_index(tools: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(tool["server_name"], tool["tool_name"]): tool for tool in tools}


def _python_literal(value: Any) -> Any:
    return repr(value) if isinstance(value, str) else value


def _api_kwargs(arguments: dict[str, Any], tool: dict[str, Any] | None) -> dict[str, Any]:
    argument_map = (tool or {}).get("argument_name_map", {})
    kwargs = {}
    for key, value in arguments.items():
        param_name = argument_map.get(key, sanitize_identifier(key))
        kwargs[param_name] = _python_literal(value)
    return kwargs


def _convert_assistant_message(content: str, tools_by_name: dict[tuple[str, str], dict[str, Any]]):
    """Convert assistant text with optional MCP XML tags into ADP actions."""
    content = content.strip()
    if not content:
        return []

    # MiroVerse constrains one MCP call per assistant turn; .search() is intentional here.
    match = MCP_CALL_RE.search(content)
    if not match:
        return [MessageAction(content=content, description=None)]

    before = content[: match.start()].strip()
    after = content[match.end() :].strip()
    block = match.group(1)
    server_name = _extract_tag(block, "server_name") or ""
    tool_name = _extract_tag(block, "tool_name") or ""
    arguments = _parse_arguments(_extract_tag(block, "arguments"))

    if isinstance(arguments, str):
        function_name = "use_mcp_tool"
        kwargs = {
            "server_name": repr(server_name),
            "tool_name": repr(tool_name),
            "arguments": repr(arguments),
        }
    else:
        tool = tools_by_name.get((server_name, tool_name))
        function_name = (tool or {}).get("function_name") or tool_function_name(
            server_name, tool_name
        )
        kwargs = _api_kwargs(arguments, tool)

    converted = [
        ApiAction(
            function=function_name,
            kwargs=kwargs,
            description=before or None,
        )
    ]
    if after:
        converted.append(MessageAction(content=after, description=None))
    return converted


def _convert_message(message, previous_was_tool_call: bool, tools_by_name):
    role = message.role
    content = message.content
    if role == "system":
        return []
    if role == "assistant":
        return _convert_assistant_message(content, tools_by_name)
    if previous_was_tool_call:
        return [TextObservation(content=content, source="environment")]
    return [TextObservation(content=content, source="user")]


def _mark_final_answer(content):
    """Wrap a plain final assistant answer with finish tags.

    MiroVerse stores terminal answers as ordinary assistant messages rather than
    explicit finish actions, so the converter marks the final message for ADP
    and OpenHands SFT compatibility.
    """
    if not content:
        return
    last = content[-1]
    if isinstance(last, MessageAction) and "<finish>" not in last.content:
        last.content = f"<finish> {last.content} </finish>"


for line in sys.stdin:
    raw_data = json.loads(line)
    data = SchemaRaw(**raw_data)
    available_tools = (
        [tool.model_dump() for tool in data.available_tools]
        if data.available_tools
        else extract_available_tools_from_messages(data.messages)
    )
    tools_by_name = _tool_index(available_tools)
    content = []
    previous_was_tool_call = False
    for message in data.messages:
        converted = _convert_message(message, previous_was_tool_call, tools_by_name)
        content.extend(converted)
        previous_was_tool_call = bool(converted) and isinstance(converted[-1], ApiAction)

    _mark_final_answer(content)

    details = {"split": data.split or ""}

    # The per-trajectory list of advertised MCP tools is recorded on the
    # top-level Trajectory.available_apis field as identifiers; the dataset's
    # api.py carries matching stubs that the OpenHands SFT converter expands
    # via include_apis when emitting the per-instance API docstring block.
    available_apis = [
        tool_function_name(tool["server_name"], tool["tool_name"]) for tool in available_tools
    ]
    # Deduplicate while preserving order.
    seen: set[str] = set()
    available_apis = [n for n in available_apis if not (n in seen or seen.add(n))]

    trajectory = create_trajectory_with_tool_call_links(
        id=data.id or f"{data.split or 'miroverse'}-unknown",
        content=content,
        available_apis=available_apis or None,
        details=details,
    )
    print(json.dumps(trajectory.model_dump(), ensure_ascii=False))
