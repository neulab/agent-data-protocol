import json
import re
import sys

from schema_raw import SchemaRaw

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.tool_call_links import create_trajectory_with_tool_call_links

FUNCTION_PATTERN = re.compile(r"<function=([^>\n]+)>\s*(.*?)\s*</function>", re.DOTALL)
PARAM_PATTERN = re.compile(r"<parameter(?:=([^>\n]+))?>(.*?)</parameter>", re.DOTALL)
POSITIONAL_PARAMS = {
    "execute_bash": ["command"],
    "str_replace_editor": ["command", "path"],
}


def clean_parameter_value(value):
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def coerce_parameter(name, value):
    value = clean_parameter_value(value)
    if name == "insert_line":
        try:
            return int(value.strip())
        except ValueError:
            return value
    if name == "view_range":
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return value


def parse_parameters(function_name, params_content):
    kwargs = {}
    positional_values = []
    for param_name, param_value in PARAM_PATTERN.findall(params_content):
        if param_name:
            kwargs[param_name] = coerce_parameter(param_name, param_value)
        else:
            positional_values.append(clean_parameter_value(param_value))

    for param_name, param_value in zip(POSITIONAL_PARAMS.get(function_name, []), positional_values):
        if param_name not in kwargs:
            kwargs[param_name] = coerce_parameter(param_name, param_value)
    return kwargs


def convert_function_call(function_name, kwargs, description, raw_call):
    if function_name == "execute_bash":
        command = kwargs.get("command")
        if command is None:
            return CodeAction(language="bash", content=raw_call, description=description)
        return CodeAction(language="bash", content=str(command), description=description)

    if function_name == "str_replace_editor":
        if "command" not in kwargs or "path" not in kwargs:
            return CodeAction(language="bash", content=raw_call, description=description)
        return ApiAction(function=function_name, kwargs=kwargs, description=description)

    if function_name == "finish":
        message = (
            kwargs.get("message") or kwargs.get("output") or description or "The task is complete."
        )
        task_completed = kwargs.get("task_completed", "true")
        if isinstance(task_completed, bool):
            task_completed = str(task_completed).lower()
        return ApiAction(
            function="finish",
            kwargs={"message": str(message), "task_completed": str(task_completed)},
        )

    return CodeAction(language="bash", content=raw_call, description=description)


def convert_assistant_message(content):
    matches = list(FUNCTION_PATTERN.finditer(content))
    if not matches:
        if "<function=" in content:
            return [CodeAction(language="bash", content=content, description=None)]
        return [MessageAction(content=content)]

    result = []
    current_pos = 0
    for match in matches:
        description = content[current_pos : match.start()].strip() or None
        function_name = match.group(1).strip()
        kwargs = parse_parameters(function_name, match.group(2))
        result.append(convert_function_call(function_name, kwargs, description, match.group(0)))
        current_pos = match.end()

    trailing_text = content[current_pos:].strip()
    if trailing_text and result:
        if result[-1].description:
            result[-1].description = f"{result[-1].description}\n\n{trailing_text}"
        else:
            result[-1].description = trailing_text
    return result


def convert_message(message):
    if message.role == "system":
        return []
    if message.role == "user":
        return [TextObservation(content=message.content, source="user")]
    if message.role == "tool":
        return [TextObservation(content=message.content, source="environment")]
    if message.role == "assistant":
        return convert_assistant_message(message.content)
    print(f"Unknown role: {message.role}", file=sys.stderr)
    return []


def process_data(data):
    content = []
    for message in data.messages:
        content.extend(convert_message(message))

    if not content:
        return None

    return create_trajectory_with_tool_call_links(
        id=data.id,
        content=content,
        details={
            "data_source": data.data_source,
        },
    )


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        if standardized_data:
            print(standardized_data.model_dump_json())
