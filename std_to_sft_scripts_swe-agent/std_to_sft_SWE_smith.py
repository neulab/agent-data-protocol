"""
This script converts a standard format to SFT format for the SWE_smith dataset, can be applied to the following datasets:
SWE-Gym_OpenHands-Sampled-Trajectories
SWE-smith_5kTrajectories
code_feedback
codeactinstruct
nebius_SWE-agent-trajectories
openhands
"""

import json
import sys
from typing import Any, Dict, List

# The system message template that was removed during conversion
SYSTEM_MESSAGE_TEMPLATE = """You are a helpful assistant that can interact with a computer to solve tasks.
<IMPORTANT>
* If user provides a path, you should NOT assume it's relative to the current working directory. Instead, you should explore the file system to find the file before working on it.
</IMPORTANT>

You have access to the following functions:

---- BEGIN FUNCTION #1: bash ----
Description: Execute a bash command in the terminal.

Parameters:
  (1) command (string, required): The bash command to execute. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.
---- END FUNCTION #1 ----

---- BEGIN FUNCTION #2: submit ----
Description: Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task.
No parameters are required for this function.
---- END FUNCTION #2 ----

---- BEGIN FUNCTION #3: str_replace_editor ----
Description: Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`

Parameters:
  (1) command (string, required): The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.
Allowed values: [`view`, `create`, `str_replace`, `insert`, `undo_edit`]
  (2) path (string, required): Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.
  (3) file_text (string, optional): Required parameter of `create` command, with the content of the file to be created.
  (4) old_str (string, optional): Required parameter of `str_replace` command containing the string in `path` to replace.
  (5) new_str (string, optional): Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.
  (6) insert_line (integer, optional): Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.
  (7) view_range (array, optional): Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.
---- END FUNCTION #3 ----


If you choose to call a function ONLY reply in the following format with NO suffix:

Provide any reasoning for the function call here.
<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Always provide reasoning for your function call in natural language BEFORE the function call (not after)
</IMPORTANT>"""

# End messages that were artificially added during standardization
# User end messages are skipped during conversion
END_USER_MESSAGES = [
    "Congratulations! You have successfully solved the task.",
    "Your solution has been verified as correct. ",
    "Well done on successfully completing the task!",
    "Your implementation satisfies the task requirements.",
    "Task completed successfully.",
]

# Assistant end messages that were artificially added during standardization
END_ASSISTANT_MESSAGES = [
    "Task completed successfully.",
    "The task has been completed.",
    "I have successfully completed the task.",
]

# Note: Assistant finish messages are now detected by pattern matching
# Any message with <finish>...</finish> pattern is converted to submit function


def format_parameter_value(value: Any) -> str:
    """Format parameter value for function call syntax."""
    if isinstance(value, list):
        return json.dumps(value)
    elif isinstance(value, dict):
        return json.dumps(value)
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, int):
        return str(value)
    else:
        return str(value)


def convert_action_to_function_call(action: Dict[str, Any]) -> str:
    """Convert an action object to function call XML syntax."""
    if action["class_"] == "code_action":
        # For bash commands
        function_call = "<function=bash>\n"
        function_call += f"<parameter=command>{action['content']}</parameter>\n"
        function_call += "</function>"
        return function_call

    elif action["class_"] == "api_action":
        function = action["function"]
        function_call = f"<function={function}>"

        # Add parameters if any
        if action.get("kwargs"):
            function_call += "\n"

            # Special handling for str_replace_editor to ensure proper parameter order
            # This maintains consistency with the function's parameter definition:
            # 1. command (required)
            # 2. path (required)
            # 3. file_text (optional, for create)
            # 4. old_str (optional, for str_replace)
            # 5. new_str (optional for str_replace, required for insert)
            # 6. insert_line (optional, for insert)
            # 7. view_range (optional, for view)
            if function == "str_replace_editor":
                # Define the parameter order for str_replace_editor
                param_order = [
                    "command",
                    "path",
                    "file_text",
                    "old_str",
                    "new_str",
                    "insert_line",
                    "view_range",
                ]

                # Add parameters in the correct order
                for param_name in param_order:
                    if param_name in action["kwargs"]:
                        param_value = action["kwargs"][param_name]
                        formatted_value = format_parameter_value(param_value)
                        function_call += f"<parameter={param_name}>{formatted_value}</parameter>\n"
            else:
                # For other functions, add parameters in any order
                for param_name, param_value in action["kwargs"].items():
                    formatted_value = format_parameter_value(param_value)
                    function_call += f"<parameter={param_name}>{formatted_value}</parameter>\n"

        function_call += "</function>"
        return function_call

    elif action["class_"] == "message_action":
        # Check if this is a finish message that should be converted to submit
        content = str(action["content"])
        if content.startswith("<finish>") and content.endswith("</finish>"):
            return "\n\n<function=submit>\n</function>"
        return content

    else:
        raise ValueError(f"Unknown action type: {action['class_']}")


def is_end_message(content_item: Dict[str, Any]) -> bool:
    """Check if this is an artificially added end message."""
    if content_item["class_"] == "text_observation":
        return content_item["content"] in END_USER_MESSAGES
    elif content_item["class_"] == "message_action":
        return content_item["content"] in END_ASSISTANT_MESSAGES
    return False


def group_content_into_messages(content: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Group content items into messages with roles."""
    messages = []

    # Add system message first
    messages.append({"role": "system", "content": SYSTEM_MESSAGE_TEMPLATE})

    i = 0
    while i < len(content):
        item = content[i]

        # Skip end user messages, but not end assistant messages
        if item["class_"] == "text_observation" and item["content"] in END_USER_MESSAGES:
            i += 1
            continue

        if item["class_"] == "text_observation":
            # Handle user/environment observations
            content_text = item["content"]
            if item["source"] == "environment":
                content_text = "OBSERVATION: " + content_text

            messages.append({"role": "user", "content": content_text})
            i += 1

        elif item["class_"] in ["code_action", "api_action", "message_action"]:
            # Group consecutive actions into one assistant message
            assistant_content_parts = []

            while i < len(content) and content[i]["class_"] in [
                "code_action",
                "api_action",
                "message_action",
            ]:
                action = content[i]

                # Add description if present
                if action.get("description"):
                    assistant_content_parts.append(action["description"])

                # Convert action to function call or message
                if action["class_"] in ["code_action", "api_action"]:
                    function_call = convert_action_to_function_call(action)
                    assistant_content_parts.append(function_call)
                else:
                    # Message action - convert or keep as is
                    message_content = convert_action_to_function_call(action)
                    assistant_content_parts.append(message_content)

                i += 1

            # Join all parts with newlines
            if assistant_content_parts:
                messages.append(
                    {"role": "assistant", "content": "\n".join(assistant_content_parts)}
                )
        else:
            # Unknown type, skip
            i += 1

    return messages


def convert_std_to_sft_SWE_smith(standardized_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert standardized format back to raw format."""
    # Extract messages from content
    messages = group_content_into_messages(standardized_data["content"])

    # Create raw format
    raw_data = {"instance_id": standardized_data["id"], "messages": messages}

    return raw_data


if __name__ == "__main__":
    # Read all input at once
    input_data = sys.stdin.read()

    # Parse the JSON data (expecting an array of standardized items)
    standardized_items = json.loads(input_data)

    # Convert each item
    raw_items = []
    for standardized_data in standardized_items:
        raw_data = convert_std_to_sft_SWE_smith(standardized_data)
        raw_items.append(raw_data)

    # Output the converted data as JSON
    print(json.dumps(raw_items, indent=2))
