import ast
import json
import re
import sys
from typing import Any, Dict, List, Union

from schema.action.action import Action
from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


def convert_function_name(function_name: str) -> str:
    """Convert function name to valid Python identifier.

    Converts hyphenated names to underscored names and removes common prefixes.
    e.g., "exa-search-web_search_exa" -> "web_search_exa"
    """
    # First replace all hyphens with underscores
    python_function_name = function_name.replace("-", "_")

    # Remove common prefixes that are redundant
    # The MCP server names are prefixed to the tool names in the raw data
    # e.g., "exa-search-web_search_exa" -> "web_search_exa"
    # Pattern: "PREFIX-actual_tool_name" where PREFIX is the server name

    # Remove "exa_search_" prefix from exa tools
    if python_function_name.startswith("exa_search_"):
        python_function_name = python_function_name[len("exa_search_") :]

    return python_function_name


def convert_tool_declarations_in_content(content: str) -> str:
    """Convert function names in tool declaration content."""

    # Pattern to match function names in tool declarations
    # e.g., "name": "exa-search-web_search_exa"
    def replace_function_name(match):
        original_name = match.group(1)
        converted_name = convert_function_name(original_name)
        return f'"name": "{converted_name}"'

    # Replace function names in JSON-like content
    pattern = r'"name":\s*"([^"]+)"'
    return re.sub(pattern, replace_function_name, content)


def _add_additional_quotes_to_strings(x):
    """
    Recursively wrap every string value with an extra pair of double quotes.
    Example: "abc" -> '"abc"'
    """
    if isinstance(x, str):
        # If you want to avoid double-wrapping already wrapped strings, uncomment:
        # if len(x) >= 2 and x[0] == '"' and x[-1] == '"':
        #     return x
        return f'"{x}"'
    if isinstance(x, list):
        return [_add_additional_quotes_to_strings(v) for v in x]
    if isinstance(x, dict):
        return {k: _add_additional_quotes_to_strings(v) for k, v in x.items()}
    return x


def parse_function_call(function_call_data: Union[dict, str]) -> ApiAction:
    """Parse function call data into ApiAction.

    Supports:
      - dict input: {"name": "...", "arguments": "..."} or {"arguments": {...}}
      - str input: "{'name': '...', 'arguments': '{\"k\": \"v\"}'}"
    """
    # NEW: accept a string that represents a Python dict
    if isinstance(function_call_data, str):
        try:
            function_call_data = ast.literal_eval(function_call_data)
        except Exception:
            # couldn't parse into dict; keep raw
            return ApiAction(
                function="unknown_function",
                kwargs={"raw_function_call_data": function_call_data},
                description=None,
            )
    function_name = function_call_data.get("name", "")
    arguments = function_call_data.get("arguments", "{}")

    if isinstance(arguments, str):
        try:
            kwargs = json.loads(arguments)
        except json.JSONDecodeError:
            kwargs = {"raw_arguments": arguments}
    else:
        kwargs = arguments

    kwargs = _add_additional_quotes_to_strings(kwargs)

    python_function_name = convert_function_name(function_name)
    return ApiAction(function=python_function_name, kwargs=kwargs, description=None)


TYPE_MAP = {
    "string": "str",
    "number": "int",
    "integer": "int",
    "boolean": "bool",
    "object": "dict",
    "array": "list",
}


DEFAULTS_BY_NAME = {
    "numResults": 5,
    "maxCharacters": 3000,
    "model": "exa-research",
    "searchType": "all",
}


def json_type_to_python(schema: Dict[str, Any]) -> str:
    if "enum" in schema:
        return "str"
    json_type = schema.get("type", "Any")
    return TYPE_MAP.get(json_type, "Any")


def generate_function_wrapper(tool: Dict[str, Any]) -> str:
    func = tool["function"]

    raw_name = func["name"]
    python_name = convert_function_name(raw_name)

    description = func.get("description", "")

    parameters = func.get("parameters", {})
    props = parameters.get("properties", {})
    required = set(parameters.get("required", []))

    arg_defs = []
    doc_lines = []

    for param_name, schema in props.items():
        py_type = json_type_to_python(schema)
        param_desc = schema.get("description", "")

        if param_name in required:
            arg_def = f"{param_name}: {py_type}"
        else:
            default_val = DEFAULTS_BY_NAME.get(param_name)

            if default_val is None:
                arg_def = f"{param_name}: Optional[{py_type}] = None"
            else:
                if isinstance(default_val, str):
                    default_val = f'"{default_val}"'
                arg_def = f"{param_name}: {py_type} = {default_val}"

        arg_defs.append(arg_def)
        doc_lines.append(f"        {param_name}: {param_desc}")

    args_signature = ", ".join(arg_defs)
    doc_args = "\n".join(doc_lines)

    return f"""
def {python_name}({args_signature}) -> dict:
    \"\"\"{description}

    Args:
    ----
{doc_args}

    \"\"\"
    pass
""".strip()


def parse_available_tools(available_tools: str) -> str:
    tools = json.loads(available_tools)

    generated_functions = []

    for tool in tools:
        generated_functions.append(generate_function_wrapper(tool))

    return "\n\n\n".join(generated_functions)


def convert_message(message: dict, message_id: str) -> List[Union[Action, Observation]]:
    """Convert a single message to standardized format."""
    role = message.get("role", "")
    content = message.get("content", "")
    reasoning = message.get("reasoning_content", "")
    function_call = message.get("function_call")

    result = []

    if role == "system":
        # Skip system messages or convert to environment observation
        return []

    elif role == "user":
        result.append(TextObservation(content=content, source="user"))

    elif role == "assistant":
        if function_call and (content.strip() or reasoning.strip()):
            api_action = parse_function_call(function_call)
            # Use reasoning_content for extended thinking, description for brief summary
            api_action.reasoning_content = reasoning if reasoning else None
            api_action.description = content if content.strip() else None
            result.append(api_action)

        # If there's a function call, create an ApiAction
        elif function_call:
            result.append(parse_function_call(function_call))

        # If there's content, create a MessageAction
        elif content.strip() or reasoning.strip():
            if content and reasoning:
                # reasoning_content for extended thinking, description stays None
                result.append(
                    MessageAction(content=content, reasoning_content=reasoning, description=None)
                )
            elif content:
                result.append(
                    MessageAction(content=content, reasoning_content=None, description=None)
                )
            elif reasoning:
                # Just reasoning, no visible content
                result.append(
                    MessageAction(content="", reasoning_content=reasoning, description=None)
                )
            else:
                raise ValueError(f"No useful information retrieved from message {message}")

        else:
            raise ValueError(f"No useful information retrieved from message {message}")

    elif role == "tool_call":
        if content.strip():
            api_action = parse_function_call(content)
            result.append(api_action)

        else:
            raise ValueError(f"No useful information retrieved from message {message}")

    elif role == "function":
        # Function results are observations from the environment
        # Convert the function name to Python identifier format
        raw_name = message.get("name", "function_result")
        converted_name = convert_function_name(raw_name)
        result.append(TextObservation(content=content, source="environment"))

    elif role == "tool_response":
        # Function results are observations from the environment
        # Convert the function name to Python identifier format
        result.append(TextObservation(content=content, source="environment"))

    return result


def combine_message_actions(prev: MessageAction, nxt: MessageAction) -> MessageAction:
    """Merge two consecutive MessageActions into one."""
    # Merge content
    content_parts = [x for x in [prev.content, nxt.content] if x]
    prev.content = "\n".join(content_parts)

    # Merge reasoning_content (extended thinking)
    reasoning_parts = []
    if hasattr(prev, "reasoning_content") and prev.reasoning_content:
        reasoning_parts.append(prev.reasoning_content)
    if hasattr(nxt, "reasoning_content") and nxt.reasoning_content:
        reasoning_parts.append(nxt.reasoning_content)
    prev.reasoning_content = "\n".join(reasoning_parts) if reasoning_parts else None

    # Merge descriptions (brief)
    desc_parts = [x for x in [prev.description, nxt.description] if x]
    prev.description = "\n".join(desc_parts) if desc_parts else None

    return prev


def combine_message_and_api_actions(message_action, api_action):
    # Combine reasoning_content from message_action if present
    if hasattr(message_action, "reasoning_content") and message_action.reasoning_content:
        if hasattr(api_action, "reasoning_content") and api_action.reasoning_content:
            api_action.reasoning_content = (
                message_action.reasoning_content + "\n" + api_action.reasoning_content
            )
        else:
            api_action.reasoning_content = message_action.reasoning_content

    # Use message content as part of description
    description = "\n" + api_action.description if api_action.description else ""
    description = message_action.content + description
    api_action.description = description if description.strip() else None
    return api_action


def interleave_api_and_text_observation(
    content: List[Union[Action, Observation]],
) -> List[Union[Action, Observation]]:
    """
    Reorder consecutive blocks of ApiAction followed by
    TextObservation into interleaved pairs.

    Example:
    [A1, A2, A3, T1, T2, T3]->[A1, T1, A2, T2, A3, T3]
    """
    new_content = []
    i = 0
    n = len(content)

    while i < n:
        # If current item is ApiAction, detect consecutive ApiAction block
        if isinstance(content[i], ApiAction):
            api_start = i
            while i < n and isinstance(content[i], ApiAction):
                i += 1
            api_end = i

            # Now check if immediately followed by TextObservation block
            text_start = i
            while i < n and isinstance(content[i], TextObservation):
                i += 1
            text_end = i

            api_block = content[api_start:api_end]
            text_block = content[text_start:text_end]

            # Only interleave if both blocks exist and lengths match
            if api_block and text_block and len(api_block) == len(text_block):
                for api, text in zip(api_block, text_block):
                    new_content.append(api)
                    new_content.append(text)
            else:
                # Fallback: keep original order
                new_content.extend(api_block)
                new_content.extend(text_block)

        else:
            new_content.append(content[i])
            i += 1

    return new_content


def convert_trajectory(raw_data: dict) -> Trajectory:
    """Convert raw Toucan data to standardized trajectory format."""
    trajectory_id = raw_data["id"]
    messages = raw_data.get("messages", [])
    available_tools = raw_data.get("available_tools", "[]")
    available_apis_doc = parse_available_tools(available_tools)
    details = {"available_apis": available_apis_doc}

    messages = json.loads(messages)

    content = []

    for i, message in enumerate(messages):
        converted_steps = convert_message(message, f"{trajectory_id}_{i}")
        if not converted_steps:
            continue
        # If consecutive message actions, combine MessageAction with previous MessageAction(s).
        while (
            content
            and converted_steps
            and isinstance(content[-1], MessageAction)
            and isinstance(converted_steps[0], MessageAction)
        ):
            content[-1] = combine_message_actions(content[-1], converted_steps[0])
            converted_steps = converted_steps[1:]

        if (
            content
            and converted_steps
            and isinstance(content[-1], MessageAction)
            and isinstance(converted_steps[0], ApiAction)
        ):
            content[-1] = combine_message_and_api_actions(content[-1], converted_steps[0])
            converted_steps = converted_steps[1:]

        content.extend(converted_steps)
    content = interleave_api_and_text_observation(content)
    # Ensure the trajectory ends with a finish action if it doesn't already
    if content and not (
        isinstance(content[-1], MessageAction) and "<finish>" in content[-1].content
    ):
        if isinstance(content[-1], MessageAction):
            # Wrap existing message action with finish tags
            content[-1].content = f"<finish> {content[-1].content} </finish>"
        else:
            # Add a finish message action
            content.append(
                MessageAction(content="<finish> Task completed. </finish>", description=None)
            )

    return Trajectory(id=trajectory_id, content=content, details=details)


# Process each line of input individually
for line in sys.stdin:
    # print(line, file=sys.stderr)
    try:
        raw_data = json.loads(line)
        standardized_trajectory = convert_trajectory(raw_data)
        print(json.dumps(standardized_trajectory.model_dump()))
    except Exception as e:
        print(f"Error processing line: {e}", file=sys.stderr)
        continue
