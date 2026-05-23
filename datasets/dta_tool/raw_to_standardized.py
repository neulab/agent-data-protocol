import ast
import json
import keyword
import re
import sys
from typing import Any

from schema_raw import SchemaRaw

from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.json import JsonObservation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

FUNCTION_CALL_RE = re.compile(r"Thought:\s*(.*?)\s*Function Call:\s*(.*)", re.DOTALL)
API_LIST_RE = re.compile(
    r"Specifically, you have access to the following APIs:\s*(\[.*\])", re.DOTALL
)


def as_python_literal(value: Any) -> str:
    return repr(value)


def parse_function_calls(message: str) -> tuple[str, list[dict[str, Any]]] | None:
    match = FUNCTION_CALL_RE.search(message)
    if not match:
        return None

    thought = match.group(1).strip()
    call_text = match.group(2).strip()
    try:
        calls = json.loads(call_text)
    except json.JSONDecodeError:
        calls = ast.literal_eval(call_text)

    if isinstance(calls, dict):
        calls = [calls]
    return thought, calls


def normalize_arguments(arguments: Any) -> dict[str, str]:
    if arguments in (None, ""):
        return {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = ast.literal_eval(arguments)
    if not isinstance(arguments, dict):
        return {}
    return {str(key): as_python_literal(value) for key, value in arguments.items()}


def extract_api_specs(system_prompt: str) -> list[dict[str, Any]]:
    match = API_LIST_RE.search(system_prompt)
    if not match:
        return []
    try:
        specs = ast.literal_eval(match.group(1).strip())
    except (SyntaxError, ValueError):
        return []
    return specs if isinstance(specs, list) else []


def parse_structured_payload(value: str) -> dict[str, Any] | list[Any] | None:
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    return None


def convert_function_observation(value: str) -> JsonObservation | TextObservation:
    parsed = parse_structured_payload(value)
    if not isinstance(parsed, dict):
        return TextObservation(content=value, source="environment")

    result = parsed.get("result")
    if isinstance(result, dict) and "response" in result:
        response = result["response"]
        if isinstance(response, str):
            response_payload = parse_structured_payload(response.strip())
            if response_payload is not None:
                return JsonObservation(content=response_payload, source="environment")
            return TextObservation(content=response, source="environment")
        if isinstance(response, (dict, list)):
            return JsonObservation(content=response, source="environment")

    return JsonObservation(content=parsed, source="environment")


def valid_identifier(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name)


def build_available_apis(system_prompt: str) -> list[str]:
    """Return the list of API function names advertised in the DTA-Tool system prompt.

    Each name is the identifier of a function declared in the dataset's `api.py`.
    The returned list is fed into ``Trajectory.available_apis`` so the OpenHands
    SFT converter can filter the API docstring block per-instance.
    """
    names: list[str] = []
    for spec in extract_api_specs(system_prompt):
        name = spec.get("name")
        if not isinstance(name, str) or name == "Finish" or not valid_identifier(name):
            continue
        if name not in names:
            names.append(name)
    return names


def convert_assistant(message: str):
    parsed = parse_function_calls(message)
    if not parsed:
        return [MessageAction(content=message, description="")]

    thought, calls = parsed
    converted = []
    for call in calls:
        name = call.get("name")
        arguments = normalize_arguments(call.get("arguments", {}))
        if name == "Finish":
            return_type = arguments.get("return_type", "'give_answer'").strip("'\"")
            answer = arguments.get("final_answer", "")
            if answer:
                try:
                    answer = ast.literal_eval(answer)
                except (SyntaxError, ValueError):
                    pass
            else:
                answer = return_type
            converted.append(
                MessageAction(content=f"<finish> {answer} </finish>", description=thought)
            )
        elif isinstance(name, str) and valid_identifier(name):
            converted.append(ApiAction(function=name, kwargs=arguments, description=thought))
        else:
            converted.append(MessageAction(content=message, description=thought))
    return converted


def process_data(raw_data: dict[str, Any]) -> Trajectory:
    data = SchemaRaw(**raw_data)
    content = []
    available_apis: list[str] = []

    for step in data.conversations:
        role = step.from_
        value = step.value
        if role == "system":
            available_apis = build_available_apis(value)
        elif role == "user":
            content.append(TextObservation(content=value, source="user"))
        elif role == "assistant":
            content.extend(convert_assistant(value))
        elif role == "function":
            content.append(convert_function_observation(value))

    return Trajectory(
        id=data.id,
        content=content,
        available_apis=available_apis or None,
        details={"source": "dongsheng/DTA-Tool"},
    )


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        try:
            standardized_data = process_data(raw_data)
        except Exception as exc:
            print(f"Skipping row due to conversion error: {exc}", file=sys.stderr)
            continue
        print(json.dumps(standardized_data.model_dump(), ensure_ascii=False))
