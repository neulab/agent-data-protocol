import json
import keyword
import re
import sys
from typing import Any

from schema_raw import Message, SchemaRaw

from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
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


def safe_identifier(name: str) -> str | None:
    if not name or not re.fullmatch(r"[A-Za-z_]\w*", name) or keyword.iskeyword(name):
        return None
    return name


def collect_called_arguments(messages: list[Message]) -> dict[str, set[str]]:
    called_args: dict[str, set[str]] = {}
    for message in messages:
        for tool_call in message.tool_calls or []:
            function_name = tool_call.function.name
            args = normalize_arguments(tool_call.function.arguments)
            called_args.setdefault(function_name, set()).update(args)
    return called_args


def build_available_apis(data: SchemaRaw) -> str:
    called_args = collect_called_arguments(data.conversations)
    definitions = []
    seen = set()

    for tool in data.tools:
        function = tool_function_definition(tool.function)
        name = function.get("name")
        safe_name = safe_identifier(name or "")
        if not safe_name or safe_name in seen:
            continue
        seen.add(safe_name)

        parameters = function.get("parameters") or {}
        properties = parameters.get("properties") or {}
        required = set(parameters.get("required") or [])
        arg_names = list(properties)
        for arg_name in sorted(called_args.get(name, set())):
            if arg_name not in arg_names:
                arg_names.append(arg_name)
                required.add(arg_name)

        signature_parts = []
        for arg_name in arg_names:
            safe_arg = safe_identifier(arg_name)
            if not safe_arg:
                continue
            signature_parts.append(safe_arg if arg_name in required else f"{safe_arg}=None")
        signature = ", ".join(signature_parts)

        description = function.get("description") or "ToolMind tool function."
        arg_docs = []
        for arg_name in arg_names:
            prop = properties.get(arg_name) or {}
            arg_description = prop.get("description") or ""
            arg_docs.append(f"    {arg_name}: {arg_description}".rstrip())
        docstring = "\n".join([description, "", "Args:", *arg_docs] if arg_docs else [description])
        definitions.append(f'def {safe_name}({signature}):\n    """{docstring}"""\n    return None')

    return "\n\n".join(definitions)


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
        if message.tool_calls:
            return [
                ApiAction(
                    function=tool_call.function.name,
                    kwargs=format_kwargs(tool_call.function.arguments),
                    description=content or None,
                )
                for tool_call in message.tool_calls
            ]
        return [MessageAction(content=content)]

    print(f"Unknown role in {role=}", file=sys.stderr)
    return []


def process_data(data: SchemaRaw) -> Trajectory:
    content = []
    for message in data.conversations:
        content.extend(message_to_events(message))

    details = {
        "source": "Nanbeige/ToolMind",
        "source_file": data.source_file,
        "row_index": str(data.row_index),
    }
    available_apis = build_available_apis(data)
    if available_apis:
        details["available_apis"] = available_apis

    return Trajectory(id=data.id, content=content, details=details)


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        print(standardized_data.model_dump_json())
