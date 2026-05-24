#!/usr/bin/env python3
"""Convert standardized format to SFT format for mini-coder dataset."""

import json
import re
import sys

ACTION_FUNCTIONS = {
    "bash": "execute_bash",
    "sh": "execute_bash",
    "shell": "execute_bash",
    "python": "execute_ipython_cell",
    "python3": "execute_ipython_cell",
}
FUNCTION_ARGS = {
    "execute_bash": "command",
    "execute_ipython_cell": "code",
}


def format_function(function_name, parameters):
    function_call = ""
    for parameter, value in parameters.items():
        function_call += f"<parameter={parameter}>\n{value}\n</parameter>\n"
    return f"<function={function_name}>\n{function_call}</function>"


def standardized_to_sft(trajectory):
    """
    Convert a standardized trajectory to SFT format.

    The SFT format has:
    - id: trajectory ID
    - system: system prompt (empty for mini-coder)
    - conversations: list of turns with "from" and "value" fields
    """
    conversations = []
    system_prompt = ""

    for item in trajectory["content"]:
        class_name = item["class_"]

        # Convert based on class type
        if class_name == "text_observation":
            if item.get("source") == "user":
                conversations.append({"from": "human", "value": item["content"]})
            elif item.get("source") == "environment":
                conversations.append({"from": "observation", "value": item["content"]})

        elif class_name == "code_action":
            # Format code action as a function call
            description = item.get("description", "")
            content = item["content"]
            language = item.get("language", "bash")
            function_name = ACTION_FUNCTIONS.get(language, "execute_ipython_cell")
            argument_name = FUNCTION_ARGS.get(function_name, "code")

            value = ""
            if description:
                value += f"{description}\n\n"
            value += format_function(function_name, {argument_name: content})

            conversations.append({"from": "function_call", "value": value})

        elif class_name == "api_action":
            # Format API action as a function call
            description = item.get("description", "")
            function = item["function"]
            args = item.get("kwargs", item.get("args", {}))

            value = ""
            if description:
                value += f"{description}\n\n"
            value += format_function(function, args)

            conversations.append({"from": "function_call", "value": value})

        elif class_name == "message_action":
            content = item["content"]
            if "<finish>" in content and "</finish>" in content:
                match = re.search(r"<finish>(.*?)</finish>", content, re.DOTALL)
                message = match.group(1).strip() if match else content
                value = format_function("finish", {"message": message, "task_completed": "true"})
                conversations.append({"from": "function_call", "value": value})
            else:
                conversations.append({"from": "gpt", "value": content})

    return {"id": trajectory["id"], "system": system_prompt, "conversations": conversations}


def main():
    """Read JSONL from stdin and write JSONL to stdout."""
    for line in sys.stdin:
        if not line.strip():
            continue

        trajectory = json.loads(line)
        sft_trajectory = standardized_to_sft(trajectory)
        print(json.dumps(sft_trajectory))


if __name__ == "__main__":
    main()
