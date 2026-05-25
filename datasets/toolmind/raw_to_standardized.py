import json
import keyword
import re
import sys
from typing import Any

from schema_raw import Message, SchemaRaw

from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.tool_call_links import create_trajectory_with_tool_call_links
from schema.trajectory import Trajectory


def python_literal(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return repr(value)


def normalize_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"arguments": arguments}
        return parsed if isinstance(parsed, dict) else {"arguments": parsed}
    if isinstance(arguments, dict):
        return arguments
    return {"arguments": arguments}


def format_kwargs(arguments: Any) -> dict[str, str]:
    return {key: python_literal(value) for key, value in normalize_arguments(arguments).items()}


def tool_function_definition(tool_function: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(tool_function, dict) or not tool_function.get("name"):
        return {}
    return tool_function


THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def safe_identifier(name: str) -> str | None:
    if not name or not re.fullmatch(r"[A-Za-z_]\w*", name) or keyword.iskeyword(name):
        return None
    return name


def split_think_blocks(content: str) -> tuple[str, str | None]:
    reasoning_parts = [part.strip() for part in THINK_BLOCK_RE.findall(content) if part.strip()]
    visible_content = THINK_BLOCK_RE.sub("", content).strip()
    reasoning_content = "\n\n".join(reasoning_parts) or None
    return visible_content, reasoning_content


def build_available_apis(data: SchemaRaw) -> list[str]:
    """Return the identifiers of the tools advertised for this ToolMind trajectory.

    The raw schema provides a `tools` array per row. Each tool's identifier is
    recorded on the top-level `Trajectory.available_apis` field so the
    OpenHands SFT converter can filter the dataset's `api.py` per-trajectory
    via `include_apis`. The full set of advertised tool stubs lives in
    `datasets/toolmind/api.py`.
    """
    names: list[str] = []
    seen: set[str] = set()
    for tool in data.tools:
        function = tool_function_definition(tool.function)
        raw_name = function.get("name") or ""
        safe_name = safe_identifier(raw_name)
        if not safe_name or safe_name in seen:
            continue
        seen.add(safe_name)
        names.append(safe_name)
    return names


def message_to_events(message: Message):
    role = message.role
    content = message.content or ""

    if role == "system":
        return [TextObservation(content=f"System instructions:\n{content}", source="user")]
    if role == "user":
        return [TextObservation(content=content, source="user")]
    if role == "tool":
        return [TextObservation(content=content, source="environment", name=message.name)]
    if role == "assistant":
        visible_content, reasoning_content = split_think_blocks(content)
        if message.tool_calls:
            return [
                ApiAction(
                    function=tool_call.function.name,
                    kwargs=format_kwargs(tool_call.function.arguments),
                    description=visible_content or None,
                    reasoning_content=reasoning_content,
                )
                for tool_call in message.tool_calls
            ]
        return [MessageAction(content=visible_content, reasoning_content=reasoning_content)]

    print(f"Unknown role in {role=}", file=sys.stderr)
    return []


def process_data(data: SchemaRaw) -> Trajectory:
    content = []
    for message in data.conversations:
        content.extend(message_to_events(message))

    details = {
        "source": "Nanbeige/ToolMind",
        "source_file": data.source_file,
        "row_index": data.row_index,
    }
    available_apis = build_available_apis(data)
    return create_trajectory_with_tool_call_links(
        id=data.id,
        content=content,
        available_apis=available_apis or None,
        details=details,
    )


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        print(standardized_data.model_dump_json())
