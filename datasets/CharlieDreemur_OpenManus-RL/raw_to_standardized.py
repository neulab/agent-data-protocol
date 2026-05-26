import ast
import json
import keyword
import re
import sys
from typing import Any

from schema_raw import SchemaRaw

from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.tool_call_links import create_trajectory_with_tool_call_links
from schema.trajectory import Trajectory

ACTION_RE = re.compile(r"(?:^|\n)(?:Action|Act)\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
THOUGHT_RE = re.compile(
    r"^\s*(?:Thought|Think)\s*:\s*(.*?)(?:\n\s*(?:Action|Act)\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)
CODE_BLOCK_RE = re.compile(r"```(?:python|json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
FINAL_ANSWER_RE = re.compile(r"Final Answer\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
FUNCTION_STYLE_ACTION_RE = re.compile(r"^([A-Za-z_][\w.]*)\((.*)\)$", re.DOTALL)
MOVIE_API_RE = re.compile(r"^Name:\s*([A-Za-z_][\w.]*)\s*\(", re.MULTILINE)
TOOLBENCH_API_RE = re.compile(r"^([A-Za-z_][\w.]*):\s+This is the subfunction", re.MULTILINE)
SINGLE_API_RE = re.compile(r"\{'name':\s*'([^']+)'.*?subfunction for tool \"([^\"]+)\"", re.DOTALL)
GOAL_RE = re.compile(r"goal:\s*(.*?)(?:\n|$)", re.IGNORECASE | re.DOTALL)
FUNCTION_REQUEST_RE = re.compile(
    r"You can input 'goal' with: '(.*)'\. Call the function ([^ ]+) "
    r"with the parameter as follows: (.*)\.",
    re.DOTALL,
)


BOILERPLATE_ACKS = {"ok", "ok.", "okay", "okay."}
NON_API_ACTIONS = {"finish"}


def to_identifier(name: str, default: str = "tool") -> str:
    identifier = re.sub(r"\W", "_", name).strip("_") or default
    if identifier[0].isdigit() or keyword.iskeyword(identifier):
        identifier = f"{default}_{identifier}"
    return identifier


def unique_identifier(name: str, used: set[str], default: str) -> str:
    base = to_identifier(name, default=default)
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def parse_structured_value(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ": "))


def canonical_message(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return canonical_json(value)
    return str(value).strip()


def normalize_tool_name(tool: Any) -> str:
    return to_identifier(str(tool), default="tool")


def kwargs_from_parameters(parameters: Any) -> dict[str, Any]:
    if parameters is None or parameters == "":
        return {}

    if isinstance(parameters, dict):
        used: set[str] = set()
        return {
            unique_identifier(str(name), used, default=f"param_{index}"): value
            for index, (name, value) in enumerate(parameters.items())
        }

    if isinstance(parameters, (list, tuple)):
        return {f"arg_{index}": value for index, value in enumerate(parameters)}

    return {"input": parameters}


def make_api_action(tool: Any, parameters: Any, reasoning_content: str | None = None) -> ApiAction:
    return ApiAction(
        function=normalize_tool_name(tool),
        kwargs=kwargs_from_parameters(parameters),
        reasoning_content=reasoning_content,
    )


def parse_json_tool_call(content: str) -> ApiAction | None:
    parsed = parse_structured_value(content)
    if not isinstance(parsed, dict):
        return None

    if "Tool" in parsed:
        tool = parsed["Tool"]
        parameters = parsed.get("Param", {})
        reasoning_content = parsed.get("goal")
    elif "Action" in parsed:
        tool = parsed["Action"]
        parameters = parsed.get("Param", parsed.get("args", {}))
        reasoning_content = parsed.get("Thought") or parsed.get("thought") or parsed.get("goal")
    elif "name" in parsed and "parameters" in parsed:
        tool = parsed["name"]
        parameters = parsed.get("parameters", {})
        reasoning_content = None
    else:
        return None

    return make_api_action(tool, parameters, reasoning_content)


def parse_code_block_tool_call(content: str) -> ApiAction | None:
    match = CODE_BLOCK_RE.search(content)
    if not match:
        return None

    parsed = parse_structured_value(match.group(1))
    if not isinstance(parsed, dict) or "name" not in parsed:
        return None

    reasoning_content = content[: match.start()].strip() or None
    return make_api_action(parsed["name"], parsed.get("parameters", {}), reasoning_content)


def extract_thought(content: str) -> str | None:
    match = THOUGHT_RE.search(content)
    if not match:
        return None
    return match.group(1).strip() or None


def parse_action_line(content: str) -> ApiAction | MessageAction | None:
    match = ACTION_RE.search(content)
    if not match:
        return None

    action_text = match.group(1).strip()
    if not action_text:
        return None

    thought = extract_thought(content)
    final_answer = FINAL_ANSWER_RE.search(action_text)
    if final_answer:
        return MessageAction(
            content=final_answer.group(1).strip(),
            reasoning_content=thought,
        )

    if " with Action Input:" in action_text:
        tool, action_input = action_text.split(" with Action Input:", 1)
        tool = tool.strip()
        parameters = parse_structured_value(action_input)
        if tool.lower() in NON_API_ACTIONS or tool.lower() == "finalaction":
            return MessageAction(
                content=canonical_message(parameters),
                reasoning_content=thought,
            )
        return make_api_action(tool, parameters, thought)

    if action_text.lower().startswith(("finish", "answer", "finalaction")):
        return MessageAction(content=action_text, reasoning_content=thought)

    function_style = FUNCTION_STYLE_ACTION_RE.match(action_text)
    if function_style:
        return make_api_action(
            function_style.group(1), {"arguments": function_style.group(2)}, thought
        )

    return ApiAction(
        function="perform_action",
        kwargs={"action": action_text},
        reasoning_content=thought,
    )


def convert_assistant(content: str) -> ApiAction | MessageAction | None:
    normalized = content.strip().lower()
    if normalized in BOILERPLATE_ACKS or "i'll follow your instructions" in normalized:
        return None

    for parser in (parse_json_tool_call, parse_code_block_tool_call, parse_action_line):
        converted = parser(content)
        if converted is not None:
            return converted

    return MessageAction(content=content)


def is_environment_observation(content: str, seen_action: bool) -> bool:
    stripped = content.strip()
    return seen_action or stripped.startswith("Observation:") or stripped.startswith('{"error"')


def normalize_observation_content(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("Observation:"):
        stripped = stripped[len("Observation:") :].strip()
    if "\nGive me one action." in stripped:
        stripped = stripped.split("\nGive me one action.", 1)[0].strip()

    parsed = parse_structured_value(stripped)
    if isinstance(parsed, dict):
        response = parsed.get("response")
        if isinstance(response, str):
            parsed_response = parse_structured_value(response)
            if not isinstance(parsed_response, str):
                parsed = {**parsed, "response": parsed_response}
        return canonical_json(parsed)
    if isinstance(parsed, list):
        return canonical_json(parsed)
    return stripped


def clean_user_content(content: str) -> str | None:
    stripped = content.strip()

    if "We detail name, description, input(parameters) and output(returns)" in stripped:
        return None
    if stripped.startswith("You have access to the following API:"):
        return None
    if "You have access to the following APIs." in stripped and "Input parameters" in stripped:
        return None

    if stripped.startswith("You should perform actions to accomplish the goal:"):
        match = GOAL_RE.search(stripped)
        if match:
            return match.group(1).strip()

    function_request = FUNCTION_REQUEST_RE.search(stripped)
    if function_request:
        goal, function_name, parameters = function_request.groups()
        return f"{goal.strip()} Call {function_name} with parameters {parameters.strip()}."

    return stripped


def parse_available_api_names(content: str) -> list[str]:
    names: list[str] = []

    for match in MOVIE_API_RE.finditer(content):
        name = match.group(1)
        if name.lower() not in NON_API_ACTIONS:
            names.append(normalize_tool_name(name))

    for match in TOOLBENCH_API_RE.finditer(content):
        names.append(normalize_tool_name(match.group(1)))

    for name, tool in SINGLE_API_RE.findall(content):
        names.append(normalize_tool_name(f"{tool}.{name}"))

    return names


def ordered_unique(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def process_data(raw_data: dict) -> Trajectory:
    data = SchemaRaw(**raw_data)
    content = []
    seen_action = False
    advertised_apis: list[str] = []
    used_apis: list[str] = []

    for turn in data.conversations:
        advertised_apis.extend(parse_available_api_names(turn.content))

        if turn.role == "assistant":
            converted = convert_assistant(turn.content)
            if converted is None:
                continue
            if isinstance(converted, ApiAction):
                seen_action = True
                used_apis.append(converted.function)
            content.append(converted)
            continue

        if turn.role in {"user", "system"}:
            source = (
                "environment" if is_environment_observation(turn.content, seen_action) else "user"
            )
            normalized_content = (
                normalize_observation_content(turn.content)
                if source == "environment"
                else clean_user_content(turn.content)
            )
            if normalized_content:
                content.append(TextObservation(content=normalized_content, source=source))
            continue

        content.append(
            TextObservation(
                content=normalize_observation_content(turn.content), source="environment"
            )
        )

    available_custom_tools = advertised_apis
    if any(api == "perform_action" for api in used_apis):
        available_custom_tools.insert(0, "perform_action")
    for api in used_apis:
        if api != "perform_action" and advertised_apis:
            available_custom_tools.append(api)

    return create_trajectory_with_tool_call_links(
        id=data.id,
        content=content,
        available_custom_tools=ordered_unique(available_custom_tools) or None,
    )


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        standardized_data = process_data(raw_data)
        print(standardized_data.model_dump_json())
