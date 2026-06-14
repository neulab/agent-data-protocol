from __future__ import annotations

import ast
import json
import keyword
import re
from typing import Any

SERVER_RE = re.compile(
    r"## Server name:\s*(?P<server>.*?)\s*\n(?P<body>.*?)(?=\n## Server name:|\Z)",
    re.DOTALL,
)
TOOL_RE = re.compile(
    r"### Tool name:\s*(?P<tool>.*?)\s*\n(?P<body>.*?)(?=\n### Tool name:|\Z)",
    re.DOTALL,
)


def sanitize_identifier(name: str) -> str:
    identifier = re.sub(r"\W", "_", name.strip())
    identifier = re.sub(r"_+", "_", identifier).strip("_")
    if not identifier:
        identifier = "value"
    if identifier[0].isdigit():
        identifier = f"tool_{identifier}"
    if keyword.iskeyword(identifier):
        identifier = f"{identifier}_"
    return identifier


def tool_function_name(server_name: str, tool_name: str) -> str:
    return f"{sanitize_identifier(server_name)}__{sanitize_identifier(tool_name)}"


def _first_mapping_literal(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    quote = None
    escaped = False
    for index, char in enumerate(text[start:], start=start):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue

        if char in {"'", '"'}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def _parse_input_schema(schema_text: str) -> dict[str, Any]:
    schema_text = _first_mapping_literal(schema_text.strip())
    if not schema_text:
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(schema_text)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, SyntaxError, TypeError, ValueError):
            continue
    return {}


def _argument_name_map(input_schema: dict[str, Any]) -> dict[str, str]:
    properties = input_schema.get("properties", {})
    used: set[str] = set()
    argument_map = {}
    for raw_name in properties:
        sanitized = sanitize_identifier(raw_name)
        candidate = sanitized
        suffix = 2
        while candidate in used:
            candidate = f"{sanitized}_{suffix}"
            suffix += 1
        used.add(candidate)
        argument_map[raw_name] = candidate
    return argument_map


def extract_available_tools_from_system_prompt(system_prompt: str) -> list[dict[str, Any]]:
    tools = []
    seen = set()
    for server_match in SERVER_RE.finditer(system_prompt):
        server_name = server_match.group("server").strip()
        for tool_match in TOOL_RE.finditer(server_match.group("body")):
            tool_name = tool_match.group("tool").strip()
            dedupe_key = (server_name, tool_name)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            block = tool_match.group("body").strip()
            if "Input JSON schema:" in block:
                description_text, schema_text = block.split("Input JSON schema:", 1)
            else:
                description_text, schema_text = block, ""
            description = re.sub(r"^Description:\s*", "", description_text.strip())
            input_schema = _parse_input_schema(schema_text)
            tools.append(
                {
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "function_name": tool_function_name(server_name, tool_name),
                    "description": description,
                    "input_schema": input_schema,
                    "argument_name_map": _argument_name_map(input_schema),
                }
            )
    return tools


def _message_value(message: Any, key: str) -> Any:
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def extract_available_tools_from_messages(messages: list[Any]) -> list[dict[str, Any]]:
    system_prompt = "\n\n".join(
        _message_value(message, "content") or ""
        for message in messages
        if _message_value(message, "role") == "system"
    )
    return extract_available_tools_from_system_prompt(system_prompt)
