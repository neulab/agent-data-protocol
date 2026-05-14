import ast
import json
import keyword
import re
import sys
from typing import Any

from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

TYPE_MAP = {
    "string": "str",
    "number": "float",
    "integer": "int",
    "boolean": "bool",
    "object": "dict",
    "array": "list",
}


def to_identifier(name: str, default: str = "tool") -> str:
    identifier = re.sub(r"\W", "_", name).strip("_") or default
    if identifier[0].isdigit() or keyword.iskeyword(identifier):
        identifier = f"{default}_{identifier}"
    return identifier


def convert_function_name(name: str) -> str:
    return to_identifier(name, default="tool")


def json_type_to_python(schema: dict[str, Any]) -> str:
    return TYPE_MAP.get(schema.get("type", "Any"), "Any")


def parse_available_apis(messages: list[dict[str, Any]]) -> str:
    functions_json = next(
        (message.get("functions") for message in messages if message.get("functions")), None
    )
    if not functions_json:
        return ""

    tools = json.loads(functions_json)
    wrappers = []
    for tool in tools:
        function = tool.get("function", {})
        name = convert_function_name(function.get("name", "tool"))
        description = function.get("description", "")
        properties = function.get("parameters", {}).get("properties", {})
        args = []
        doc_lines = []
        for raw_param, param_schema in properties.items():
            param = to_identifier(raw_param, default="param")
            args.append(f"{param}: {json_type_to_python(param_schema)} | None = None")
            param_description = param_schema.get("description", "")
            doc_lines.append(f"        {param}: {param_description}")

        signature = ", ".join(args)
        args_doc = "\n".join(doc_lines) if doc_lines else "        None"
        wrappers.append(
            f"def {name}({signature}) -> dict:\n"
            f'    """{description}\n\n'
            f"    Args:\n"
            f"    ----\n"
            f"{args_doc}\n\n"
            f'    """\n'
            f"    pass"
        )
    return "\n\n\n".join(wrappers)


def function_name_from_ast(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = function_name_from_ast(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ast.unparse(node)


def quote_strings(value: Any) -> Any:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return [quote_strings(item) for item in value]
    if isinstance(value, tuple):
        return [quote_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: quote_strings(item) for key, item in value.items()}
    return value


def literal_or_source(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node)


def parse_function_call(call: str) -> ApiAction:
    expression = ast.parse(call.strip(), mode="eval").body
    if not isinstance(expression, ast.Call):
        raise ValueError(f"Function call is not an ast.Call: {call}")

    kwargs = {
        to_identifier(keyword_arg.arg, default="param"): quote_strings(
            literal_or_source(keyword_arg.value)
        )
        for keyword_arg in expression.keywords
        if keyword_arg.arg is not None
    }
    for index, arg in enumerate(expression.args):
        kwargs[f"arg_{index}"] = quote_strings(literal_or_source(arg))

    return ApiAction(
        function=convert_function_name(function_name_from_ast(expression.func)),
        kwargs=kwargs,
        description=None,
    )


def split_function_calls(function_calls: str) -> list[str]:
    return [line.strip() for line in function_calls.splitlines() if line.strip()]


def split_environment_content(content: str) -> list[TextObservation]:
    parts = [line for line in content.splitlines() if line.strip()]
    if not parts:
        parts = [content]
    return [TextObservation(content=part, source="environment") for part in parts]


def interleave_api_actions_and_observations(content: list[Any]) -> list[Any]:
    interleaved = []
    index = 0
    while index < len(content):
        if not isinstance(content[index], ApiAction):
            interleaved.append(content[index])
            index += 1
            continue

        api_start = index
        while index < len(content) and isinstance(content[index], ApiAction):
            index += 1
        api_block = content[api_start:index]

        observation_start = index
        while (
            index < len(content)
            and isinstance(content[index], TextObservation)
            and content[index].source == "environment"
        ):
            index += 1
        observation_block = content[observation_start:index]

        if len(api_block) == len(observation_block) and observation_block:
            for api_action, observation in zip(api_block, observation_block):
                interleaved.extend([api_action, observation])
        else:
            interleaved.extend(api_block)
            interleaved.extend(observation_block)
    return interleaved


def convert_message(message: dict[str, Any]) -> list[Any]:
    role = message["role"]
    content = message.get("content")
    function_calls = message.get("function_calls")

    if role == "system":
        return []
    if role == "user":
        return [TextObservation(content=content or "", source="user")]
    if role == "environment":
        return split_environment_content(content or "")
    if role == "assistant":
        if function_calls:
            return [parse_function_call(call) for call in split_function_calls(function_calls)]
        if content:
            return [MessageAction(content=content, description=None)]
        return []
    raise ValueError(f"Unexpected role: {role}")


def convert_trajectory(raw_data: dict[str, Any]) -> Trajectory:
    messages = raw_data["messages"]
    content = []
    for message in messages:
        content.extend(convert_message(message))

    content = interleave_api_actions_and_observations(content)
    if content and isinstance(content[-1], MessageAction):
        content[-1].content = f"<finish> {content[-1].content} </finish>"
    elif content:
        content.append(
            MessageAction(content="<finish> Task completed. </finish>", description=None)
        )

    details = {
        "dataset_source": raw_data.get("dataset_source", ""),
        "available_apis": parse_available_apis(messages),
    }
    return Trajectory(id=raw_data["id"], content=content, details=details)


for line in sys.stdin:
    try:
        raw = json.loads(line)
        print(json.dumps(convert_trajectory(raw).model_dump(), ensure_ascii=False))
    except Exception as exc:
        print(f"Error processing row: {exc}", file=sys.stderr)
