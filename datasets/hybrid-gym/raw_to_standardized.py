import json
import re
import sys
from typing import Any

from schema_raw import Message, SchemaRaw

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.tool_call_links import create_trajectory_with_tool_call_links

FUNCTION_RE = re.compile(r"<function=([^>\n]+)>\s*(.*?)\s*</function>", re.DOTALL)
PARAMETER_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)


def _trim_parameter_value(value: str) -> str:
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def parse_function_parameters(function_body: str) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for match in PARAMETER_RE.finditer(function_body):
        name = match.group(1).strip()
        value = _trim_parameter_value(match.group(2))
        if name == "insert_line":
            try:
                parameters[name] = int(value)
                continue
            except ValueError:
                pass
        if name == "view_range":
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    parameters[name] = parsed
                    continue
            except json.JSONDecodeError:
                pass
        parameters[name] = value
    return parameters


def function_call_to_event(function_name: str, parameters: dict[str, Any], description: str | None):
    if function_name == "execute_bash":
        if "command" not in parameters:
            print("WARNING: execute_bash missing command parameter", file=sys.stderr)
        else:
            return CodeAction(
                language="bash",
                content=str(parameters["command"]),
                description=description,
            )
    if function_name == "finish":
        message = str(parameters.get("message") or parameters.get("content") or "").strip()
        return MessageAction(content=f"<finish> {message} </finish>", description=description)
    if function_name in {"str_replace_editor", "think"}:
        return ApiAction(function=function_name, kwargs=parameters, description=description)
    return MessageAction(
        content=format_function_call(function_name, parameters),
        description=description,
    )


def format_function_call(function_name: str, parameters: dict[str, Any]) -> str:
    parameter_text = "".join(
        f"<parameter={name}>\n{value}\n</parameter>\n" for name, value in parameters.items()
    )
    return f"<function={function_name}>\n{parameter_text}</function>"


def convert_assistant_message(content: str):
    events = []
    cursor = 0
    for match in FUNCTION_RE.finditer(content):
        prefix = content[cursor : match.start()].strip()
        if prefix and events:
            events.append(MessageAction(content=prefix))
            description = None
        else:
            description = prefix or None
        function_name = match.group(1).strip()
        parameters = parse_function_parameters(match.group(2))
        events.append(function_call_to_event(function_name, parameters, description))
        cursor = match.end()

    suffix = content[cursor:].strip()
    if suffix:
        events.append(MessageAction(content=suffix))
    if events:
        return events
    return [MessageAction(content=content)] if content.strip() else []


def convert_message(message: Message, previous_event):
    if message.role == "system":
        return []
    if message.role == "assistant":
        return convert_assistant_message(message.content)
    source = "environment" if isinstance(previous_event, (ApiAction, CodeAction)) else "user"
    return [TextObservation(content=message.content, source=source)]


def process_data(data: SchemaRaw):
    content = []
    previous_event = None
    for message in data.messages:
        converted = convert_message(message, previous_event)
        content.extend(converted)
        if converted:
            previous_event = converted[-1]

    return create_trajectory_with_tool_call_links(
        id=data.id,
        content=content,
        details={
            "source_dataset": data.source_dataset,
            "task_type": data.task_type,
            "split": data.split,
            "row_index": data.row_index,
        },
    )


def main():
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        print(json.dumps(standardized_data.model_dump(), ensure_ascii=False))


if __name__ == "__main__":
    main()
