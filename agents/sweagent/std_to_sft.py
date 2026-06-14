import json
import os
import re
import sys

from agents.sweagent.api import get_api_tool_description
from agents.sweagent.system_message import base_template
from scripts.atif_input import (
    ApiAction,
    CodeAction,
    MessageAction,
    TextObservation,
    WebObservation,
    load_trajectory,
)

dataset = os.getenv("MY_DATASET")
assert dataset, "Please set the environment variable MY_DATASET"

sweagent_default_tools = {
    "bash": {"required": ["command"], "optional": []},
    "submit": {"required": [], "optional": []},
    "str_replace_editor": {
        "required": ["command", "path"],
        "optional": ["file_text", "old_str", "new_str", "insert_line", "view_range"],
    },
}
HF_ROLE_MAP = {
    "human": "user",
    "observation": "user",
    "gpt": "assistant",
    "function_call": "assistant",
}


def to_hf_messages(conversations: list[dict]) -> list[dict]:
    messages = []
    for message in conversations:
        role = HF_ROLE_MAP.get(message["from"])
        if role is None:
            raise ValueError(f"Cannot convert SFT role to Hugging Face role: {message['from']}")
        messages.append({"role": role, "content": message["value"]})
    return messages


def get_system_message(api_tool_description) -> str:
    """
    Generate the system message template, optionally including dataset-specific API functions.
    """
    # Get dataset-specific API functions if dataset_name is provided
    system_message = base_template
    return system_message.strip()


# Convert function call to OH format
def format_function(function_name, parameters):
    # Example OH function format:
    """
    <function=example_function_name>
    <parameter=example_parameter_1>value_1</parameter>
    <parameter=example_parameter_2>
    This is the value for the second parameter
    that can span
    multiple lines
    </parameter>
    </function>
    """

    function_call = ""
    for parameter in parameters:
        value = parameters[parameter]
        function_call += f"<parameter={parameter}>\n{value}\n</parameter>\n"
    function_call = f"<function={function_name}>\n{function_call}</function>"
    return function_call


def verify_args(required_args, optional_args, input_args):
    # all required args should be included
    for arg in required_args:
        if arg not in input_args:
            return False
    # all input args should be one of the specified args of the function
    for arg in input_args:
        if arg not in required_args + optional_args:
            return False
    return True


def standardized_event_to_swe_message(
    id,
    event: ApiAction | CodeAction | MessageAction | TextObservation | WebObservation,
    api_sigs=None,
) -> dict:
    if isinstance(event, ApiAction):
        thought = f"<think>\n{event.description}\n</think>\n\n" if event.description else ""
        function_name = event.function
        arguments = {k: v for k, v in event.kwargs.items() if k not in ["element_id", "xpath"]}
        if function_name == "file_editor":
            function_name = "str_replace_editor"

        # for tool that are one of the default SWE-Agent tools
        if function_name in sweagent_default_tools and function_name not in api_sigs:
            tool_args = sweagent_default_tools[function_name]
            if not verify_args(tool_args["required"], tool_args["optional"], arguments):
                raise ValueError(f"Function call with wrong argument: {event}")
            function_call = format_function(function_name, arguments)
            return {"from": "function_call", "value": f"{thought}{function_call}"}

        # these should all be dataset specific apis, and should be called in bash
        if function_name in api_sigs:
            if not verify_args(
                api_sigs[function_name]["required"], api_sigs[function_name]["optional"], arguments
            ):
                raise ValueError(f"Function call with wrong argument: {event}")
            api_action = f"{function_name}({', '.join([f'{k}={arguments[k]}' for k in arguments])})"
            function_call = format_function("bash", {"command": api_action})
            return {"from": "function_call", "value": f"{thought}{function_call}"}

        if function_name == "think":
            thought_text = arguments.get("thought") or arguments.get("content") or ""
            function_call = format_function("bash", {"command": f"think(thought={thought_text!r})"})
            return {"from": "function_call", "value": f"{thought}{function_call}"}

        raise ValueError(f"Undefined API: {event}")

    if isinstance(event, CodeAction):
        thought = f"<think>\n{event.description}\n</think>\n\n" if event.description else ""
        code_content = event.content
        if event.language != "bash":
            if event.language == "python" or event.language == "python3":
                # python -c $'import math\nprint(math.sqrt(2))\nfor i in range(3):\n    print(i)'
                code_content = f"{event.language} -c $'{code_content}'"
            else:
                return None
        code_action = format_function("bash", {"command": code_content})
        return {"from": "function_call", "value": f"{thought}{code_action}"}

    elif isinstance(event, MessageAction):
        thought = f"<think>\n{event.description}\n</think>\n\n" if event.description else ""
        # convert finish actions to submit actions
        if "<finish>" in event.content and "</finish>" in event.content:
            match = re.search(r"<finish>(.*?)</finish>", event.content, re.DOTALL)
            content = match.group(1).strip()
            thought += f"\n{content}"
            submit_function_call = format_function("submit", {})
            return {"from": "function_call", "value": f"{thought.strip()}{submit_function_call}"}
        return {"from": "gpt", "value": f"{thought}{event.content}"}

    elif isinstance(event, TextObservation):
        if event.source == "user":
            event.source = "human"

        elif event.source == "agent":
            event.source = "gpt"

        elif event.source == "environment":
            event.source = "observation"

        else:
            raise ValueError(f"Wrong event source: {event.source}")
        return {"from": event.source, "value": event.content}

    else:
        raise ValueError(f"Unknown event type: {type(event)}\n{event}")


def process_row(line, api_tool_description, api_sigs):
    trajectory = load_trajectory(line)
    id = trajectory.id
    events = trajectory.content
    if trajectory.available_apis is not None:
        try:
            api_tool_description, api_sigs = get_api_tool_description(
                dataset, include_apis=trajectory.available_apis
            )
        except Exception as e:
            print(e, file=sys.stderr)
            return None
    conversations = []
    for i in range(len(events)):
        event = events[i]
        try:
            message = standardized_event_to_swe_message(id, event, api_sigs)
            if not message:
                print(event, file=sys.stderr)
                return None
            if conversations and (
                conversations[-1]["from"] == "function_call" or message["from"] == "observation"
            ):
                message["value"] = "OBSERVATION:\n" + message["value"]
            conversations.extend([message])

        except Exception as e:
            # traceback.print_exc()
            print(e, file=sys.stderr)
            return None
    for message in conversations:
        if message["from"] == "function_call":
            message["from"] = "gpt"
        elif message["from"] == "observation":
            message["from"] = "human"
    return {
        "id": trajectory.id,
        "conversations": to_hf_messages(conversations),
        "system": get_system_message(api_tool_description),
    }


def main():
    api_tool_description, api_sigs = get_api_tool_description(dataset)
    for line in sys.stdin:
        output_line = process_row(
            line,
            api_tool_description=api_tool_description,
            api_sigs=api_sigs,
        )
        if output_line:
            assert json.loads(json.dumps(output_line))
            print(json.dumps(output_line))
            # with open(f"datasets/{dataset}/full_sft_swe.jsonl", "a") as f:
            #     try:
            #         f.write(json.dumps(output_line) + "\n")
            #     except Exception as e:
            #         traceback.print_exc()
            #         print(e)
            #         continue


if __name__ == "__main__":
    main()
