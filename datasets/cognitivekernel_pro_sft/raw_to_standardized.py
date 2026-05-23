#!/usr/bin/env python3
"""Convert CognitiveKernel-Pro-SFT records to ADP standardized trajectories."""

import ast
import json
import re
import sys
from typing import Any

from schema_raw import SchemaRaw

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

THOUGHT_CODE_RE = re.compile(
    r"^\s*Thought:\s*(?P<thought>.*?)\s*\nCode:\s*```(?:python)?\s*\n(?P<code>.*?)\n```\s*$",
    re.DOTALL,
)
API_ARGUMENTS = {
    "web_agent": ["task"],
    "file_agent": ["task", "file_path_dict"],
    "stop": ["output", "log"],
    "ask_llm": ["query"],
    "simple_web_search": ["query"],
    "load_file": ["file_name"],
    "read_text": ["file_name", "page_id_list"],
    "read_screenshot": ["file_name", "page_id_list"],
    "search": ["file_name", "key_word_list"],
}
TABLEBENCH_API_ARGUMENTS = {**API_ARGUMENTS, "stop": ["answer", "summary"]}
AVAILABLE_APIS_BY_SOURCE_FILE = {
    "tablebench.sft.jsonl": [
        "load_file",
        "read_text",
        "read_screenshot",
        "search",
        "stop",
    ],
    "ck-pro-web.sft.jsonl": [
        "web_agent",
        "file_agent",
        "stop",
        "ask_llm",
        "simple_web_search",
    ],
    "docbench.sft.jsonl": [
        "web_agent",
        "file_agent",
        "stop",
        "ask_llm",
        "simple_web_search",
    ],
    "webwalker_subset.sft.jsonl": [
        "web_agent",
        "file_agent",
        "stop",
        "ask_llm",
        "simple_web_search",
    ],
}


def function_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = function_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def literal_or_source(
    node: ast.AST, variables: dict[str, tuple[Any, bool]]
) -> tuple[Any, bool]:
    if isinstance(node, ast.Name) and node.id in variables:
        return variables[node.id]
    try:
        return ast.literal_eval(node), True
    except Exception:
        return ast.unparse(node), False


def format_kwarg_value(value: Any, is_literal: bool) -> Any:
    if isinstance(value, str) and is_literal:
        return json.dumps(value, ensure_ascii=False)
    return value


def collect_literal_assignments(tree: ast.Module) -> dict[str, tuple[Any, bool]]:
    variables: dict[str, tuple[Any, bool]] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        value = literal_or_source(statement.value, variables)
        for target in statement.targets:
            if isinstance(target, ast.Name):
                variables[target.id] = value
    return variables


def parse_api_action(
    code: str, description: str, api_arguments: dict[str, list[str]]
) -> ApiAction | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    variables = collect_literal_assignments(tree)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and function_name(node.func) in api_arguments
    ]
    if len(calls) != 1:
        return None

    call = calls[0]
    name = function_name(call.func)
    if name is None:
        return None

    kwargs = {}
    positional_names = api_arguments[name]
    for index, arg in enumerate(call.args):
        arg_name = positional_names[index] if index < len(positional_names) else f"arg_{index}"
        value, is_literal = literal_or_source(arg, variables)
        kwargs[arg_name] = format_kwarg_value(value, is_literal)

    for keyword in call.keywords:
        if keyword.arg is None:
            return None
        value, is_literal = literal_or_source(keyword.value, variables)
        kwargs[keyword.arg] = format_kwarg_value(value, is_literal)

    return ApiAction(function=name, kwargs=kwargs, description=description)


def api_arguments_for_source(source_file: str) -> dict[str, list[str]]:
    if source_file == "tablebench.sft.jsonl":
        return TABLEBENCH_API_ARGUMENTS
    return API_ARGUMENTS


def convert_assistant_message(
    content: str, source_file: str
) -> ApiAction | CodeAction | MessageAction:
    match = THOUGHT_CODE_RE.match(content)
    if not match:
        # Released samples follow Thought/Code; this preserves unexpected rows.
        return MessageAction(content=content)

    code = match.group("code").strip()
    thought = match.group("thought").strip()
    api_action = parse_api_action(code, thought, api_arguments_for_source(source_file))
    if api_action is not None:
        return api_action

    return CodeAction(language="python", content=code, description=thought)


def process_record(record: SchemaRaw) -> Trajectory:
    user_messages = [message.content for message in record.messages if message.role == "user"]
    assistant_messages = [
        message.content for message in record.messages if message.role == "assistant"
    ]

    if not user_messages:
        raise ValueError(f"Record {record.id} has no user message")
    if not assistant_messages:
        raise ValueError(f"Record {record.id} has no assistant message")

    content = [TextObservation(content="\n\n".join(user_messages), source="user")]
    content.extend(
        convert_assistant_message(message, record.source_file)
        for message in assistant_messages
    )

    return Trajectory(
        id=record.id,
        content=content,
        available_apis=AVAILABLE_APIS_BY_SOURCE_FILE.get(record.source_file),
        details={
            "source": "CognitiveKernel/CognitiveKernel-Pro-SFT",
            "source_file": record.source_file,
            "source_index": str(record.source_index),
        },
    )


if __name__ == "__main__":
    for line in sys.stdin:
        raw_record = SchemaRaw(**json.loads(line))
        print(process_record(raw_record).model_dump_json())
