import json
import re
import sys
from typing import List, Union

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


def parse_function_call(function_call_data: dict) -> ApiAction:
    """Parse function call data into ApiAction."""
    function_name = function_call_data.get("name", "")
    arguments = function_call_data.get("arguments", "{}")

    # Parse arguments if they're a string
    if isinstance(arguments, str):
        try:
            kwargs = json.loads(arguments)
        except json.JSONDecodeError:
            kwargs = {"raw_arguments": arguments}
    else:
        kwargs = arguments

    python_function_name = convert_function_name(function_name)

    return ApiAction(function=python_function_name, kwargs=kwargs, description=None)


def convert_message(message: dict, message_id: str) -> List[Union[Action, Observation]]:
    """Convert a single message to standardized format."""
    role = message.get("role", "")
    content = message.get("content", "")
    function_call = message.get("function_call")

    result = []

    if role == "system":
        # Skip system messages or convert to environment observation
        if content.strip():
            # Convert function names in tool declarations within system message content
            converted_content = convert_tool_declarations_in_content(content)
            result.append(
                TextObservation(content=converted_content, source="environment", name="system")
            )

    elif role == "user":
        result.append(TextObservation(content=content, source="user"))

    elif role == "assistant":
        # If there's a function call, create an ApiAction
        if function_call:
            result.append(parse_function_call(function_call))

        # If there's content, create a MessageAction
        if content.strip():
            result.append(MessageAction(content=content, description=None))

    elif role == "function":
        # Function results are observations from the environment
        # Convert the function name to Python identifier format
        raw_name = message.get("name", "function_result")
        converted_name = convert_function_name(raw_name)
        result.append(TextObservation(content=content, source="environment", name=converted_name))

    return result


def convert_trajectory(raw_data: dict) -> Trajectory:
    """Convert raw Toucan data to standardized trajectory format."""
    trajectory_id = raw_data["id"]
    messages = raw_data.get("messages", [])

    content = []

    for i, message in enumerate(messages):
        converted_steps = convert_message(message, f"{trajectory_id}_{i}")
        content.extend(converted_steps)

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

    return Trajectory(id=trajectory_id, content=content)


# Process each line of input individually
for line in sys.stdin:
    try:
        raw_data = json.loads(line)
        standardized_trajectory = convert_trajectory(raw_data)
        print(json.dumps(standardized_trajectory.model_dump()))
    except Exception as e:
        print(f"Error processing line: {e}", file=sys.stderr)
        continue
