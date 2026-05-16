#!/usr/bin/env python3
"""Convert raw SWE-Play-trajectories data to ADP standardized format."""

import json
import re
import sys
from typing import Optional

from schema_raw import SchemaRaw

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


def parse_function_call(content: str) -> tuple[Optional[str], dict, str]:
    """Parse a function call from assistant message content.

    Returns:
        tuple: (function_name, kwargs, description)
            - function_name: The name of the function called, or None if no function call
            - kwargs: Dictionary of parameter names to values
            - description: The text before the function call
    """
    # Match function call pattern: <function=function_name>..parameters..</function>
    func_pattern = r"<function=(\w+)>(.*?)</function>"
    func_match = re.search(func_pattern, content, re.DOTALL)

    if not func_match:
        return None, {}, content.strip()

    function_name = func_match.group(1)
    params_block = func_match.group(2)

    # Get the description (text before the function call)
    description = content[: func_match.start()].strip()

    # Parse parameters: <parameter=param_name>value</parameter>
    param_pattern = r"<parameter=(\w+)>(.*?)</parameter>"
    params = re.findall(param_pattern, params_block, re.DOTALL)

    kwargs = {}
    for param_name, param_value in params:
        # Clean up the value
        value = param_value.strip()
        # Try to convert to appropriate type
        if value.isdigit():
            value = int(value)
        elif value.lower() in ("true", "false"):
            value = value.lower() == "true"
        kwargs[param_name] = value

    return function_name, kwargs, description


def convert_assistant_message(content: str) -> list:
    """Convert an assistant message to ADP actions.

    Args:
        content: The assistant message content

    Returns:
        list: List of action objects
    """
    actions = []
    function_name, kwargs, description = parse_function_call(content)

    if function_name is None:
        # No function call, just a message
        actions.append(MessageAction(content=content.strip(), description=None))
    elif function_name == "execute_bash":
        # Bash code execution
        code = kwargs.get("command", "")
        actions.append(
            CodeAction(
                language="bash",
                content=code,
                description=description if description else None,
            )
        )
    elif function_name == "execute_ipython_cell":
        # Python code execution
        code = kwargs.get("code", "")
        actions.append(
            CodeAction(
                language="python",
                content=code,
                description=description if description else None,
            )
        )
    elif function_name in ("str_replace_editor", "think", "finish"):
        # API actions
        actions.append(
            ApiAction(
                function=function_name,
                kwargs=kwargs,
                description=description if description else None,
            )
        )
    else:
        # Unknown function, treat as API action
        actions.append(
            ApiAction(
                function=function_name,
                kwargs=kwargs,
                description=description if description else None,
            )
        )

    return actions


def convert_raw_to_standardized(raw_data: SchemaRaw) -> Trajectory:
    """Convert raw data to ADP standardized format.

    Args:
        raw_data: The raw data from HuggingFace

    Returns:
        Trajectory: The standardized trajectory
    """
    content = []
    system_prompt = None
    is_first_user_message = True

    for msg in raw_data.messages:
        if msg.role == "system":
            # Store system prompt for details
            system_prompt = msg.content
        elif msg.role == "user":
            if is_first_user_message:
                # First user message is the task description
                content.append(
                    TextObservation(
                        content=msg.content,
                        source="user",
                    )
                )
                is_first_user_message = False
            else:
                # Subsequent user messages are execution results
                content.append(
                    TextObservation(
                        content=msg.content,
                        source="environment",
                    )
                )
        elif msg.role == "assistant":
            # Parse assistant message for actions
            actions = convert_assistant_message(msg.content)
            content.extend(actions)

    return Trajectory(
        id=raw_data.id,
        content=content,
        details={
            "category": raw_data.category or "",
            "source": raw_data.source or "",
        },
    )


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized = convert_raw_to_standardized(data)
        print(standardized.model_dump_json())
