import argparse
import json
import os
import re
import sys
import traceback

# Removed unused imports
from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory
from scripts.api import get_api_tool_description
from scripts.html_to_axtree import HTMLToAXTree
from scripts.system_prompt.system import get_system_message
from scripts.system_prompt.user import get_web_user_message

dataset = os.getenv("MY_DATASET")
assert dataset, "Please set the environment variable MY_DATASET"

openhands_default_tools = [
    "execute_bash",
    "think",
    "finish",
    "web_read",
    "browser",
    "execute_ipython_cell",
    "str_replace_editor",
    "edit_file",
]

action_function = {"python": "execute_ipython_cell", "bash": "execute_bash", "web": "browser"}

function_args = {
    "execute_ipython_cell": "code",
    "execute_bash": "command",
}

USE_NAV = (
    os.environ.get("USE_NAV", "true") == "true"
)  # only disable NAV actions when running webarena and miniwob benchmarks

generate_axtree = HTMLToAXTree(dataset)

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


def format_function(function, parameters):
    if function not in openhands_default_tools:
        return None
    function_call = ""
    for parameter in parameters:
        value = parameters[parameter]
        function_call += f"<parameter={parameter}>\n{value}\n</parameter>\n"
    function_call = f"<function={function}>\n{function_call}</function>"
    return function_call


def extract_function_call(content):
    for tool in openhands_default_tools:
        if f"<function={tool}" in content:
            return tool
    return None


def standardized_event_to_openhands_message(
    id,
    event: ApiAction | CodeAction | MessageAction | TextObservation | WebObservation,
    details: dict,
    previous_web_actions: list,
    is_web: str,
    chunk: str,
    api_env: str = None,
) -> dict:
    # NOTE for KETAN: deal with the different types of events later
    # The Web and API Actions are based on Browsergym's schema. So use normal actions if the style is different to HTML/AXTree
    if isinstance(event, WebObservation):
        if event.axtree is not None:
            axtree = event.axtree
        elif generate_axtree.last_html != event.html:
            axtree = generate_axtree.build_axtree(id, event.html, chunk)
        else:
            axtree = generate_axtree.last_xtree
        prompt = get_web_user_message("", event.url, axtree, previous_web_actions)
        return {"from": "human", "value": prompt}

    if isinstance(event, ApiAction):
        thought = event.description + "\n\n" if event.description else ""

        if (
            event.function == "goto"
        ):  # could add more or conditions here for actions that don't require bid
            api_action = f"{event.function}({', '.join([f'{k}={v}' for k, v in event.kwargs.items() if k not in ['element_id', 'xpath']])})"
            previous_web_actions.extend([api_action])
            call = json.loads(f'{{"name": "browser", "arguments": {{"code": "{api_action}"}}}}')
            function_call = format_function(call["name"], call["arguments"])
            return {"from": "function_call", "value": f"{thought}{function_call}"}

        arguments = None
        # try to directly get the browsergym_id from the event kwargs
        browsergym_id = event.kwargs.get("element_id", None)

        # this gets the browsergym_id of the element that the user is interacting with
        # the latest(last seen) html's obs is updated whenever build_axtree is called
        # the latest obs is used to get the browsergym_id
        if not browsergym_id:
            event_xpath = event.kwargs.get("xpath", None)
            if event_xpath:
                browsergym_id = generate_axtree.get_bid(id, event_xpath, chunk)
        # for tool calls that are not browser based
        if not browsergym_id and event.function in openhands_default_tools:
            arguments = {k: v for k, v in event.kwargs.items() if k not in ["element_id", "xpath"]}
            function_call = format_function(event.function, arguments)
            return {"from": "function_call", "value": f"{thought}{function_call}"}
        if not browsergym_id:
            local_api_env = api_env
            if not local_api_env:
                # Default to 'execute_bash' if api_env is not specified
                local_api_env = "execute_bash"
            arg = function_args.get(local_api_env, "code")
            api_action = f"{event.function}({', '.join([f'{k}={v}' for k, v in event.kwargs.items() if k not in ['element_id', 'xpath']])})"
            function_call = format_function(local_api_env, {arg: api_action})
            return {"from": "function_call", "value": f"{thought}{function_call}"}
        # for tool calls that are browser based
        elif len(event.kwargs) == 1 and "element_id" in event.kwargs:
            api_action = f"{event.function}(bid={browsergym_id})"
        else:
            api_action = f"{event.function}(bid={browsergym_id}, {', '.join([f'{k}={v}' for k, v in event.kwargs.items() if k not in ['element_id', 'xpath']])})"
        previous_web_actions.extend([api_action])
        call = json.loads(f'{{"name": "browser", "arguments": {{"code": "{api_action}"}}}}')
        call = format_function(call["name"], call["arguments"])
        return {"from": "function_call", "value": f"{thought}{call}"}

    if isinstance(event, CodeAction):
        thought = event.description + "\n\n" if event.description else ""
        function_name = action_function.get(event.language, f"execute_{event.language}")
        arg = function_args.get(function_name, "code")
        code_action = format_function(function_name, {arg: event.content})
        if not code_action:
            raise ValueError(f"Event with unknown code action type: {type(event)}\n{function_name}")
        return {"from": "function_call", "value": f"{thought}{code_action}"}

    elif isinstance(event, MessageAction):
        thought = event.description + "\n\n" if event.description else ""
        if "<finish>" in event.content and "</finish>" in event.content:
            match = re.search(r"<finish>(.*?)</finish>", event.content, re.DOTALL)
            content = match.group(1).strip()
            finish_function_call = format_function(
                "finish", {"message": content, "task_completed": "true"}
            )
            return {"from": "function_call", "value": f"{thought}{finish_function_call}"}
        return {"from": "gpt", "value": f"{thought}{event.content}"}

    elif isinstance(event, TextObservation):
        # I had this earlier to include source in the message, but OpenHands does not have that and has bash executions as user messages
        # return {"role": event.source, "content": event.content} if event.source == "user" or event.source=='system' else {"role": "user", "content": f"OBSERVATION from {event.source}: {event.content}"}

        # Check if this is a system message containing function descriptions
        if event.source == "system" and (
            "<function=" in event.content
            or "<function_calls>" in event.content
            or "<invoke name=" in event.content
        ):
            return {"from": "function_call", "value": event.content}

        if event.source == "user":
            event.source = "human"

        if event.source == "assistant":
            event.source = "gpt"

        if event.source == "system":
            return {"from": "human", "value": f"{event.content}"}
        else:
            return {"from": event.source, "value": event.content}

    elif hasattr(event, "__class__") and event.__class__.__name__ == "ImageObservation":
        # Handle ImageObservation
        annotations_text = ""
        if hasattr(event, "annotations") and event.annotations:
            annotations = []
            for annotation in event.annotations:
                if hasattr(annotation, "text") and annotation.text:
                    annotations.append(f"{annotation.text} ({annotation.element_type})")
            if annotations:
                annotations_text = "Elements detected: " + ", ".join(annotations)

        image_path = getattr(event, "content", "unknown_image_path")
        return {"from": "observation", "value": f"[Image: {image_path}]\n{annotations_text}"}

    else:
        raise ValueError(f"Unknown event type: {type(event)}\n{event}")


def process_row(line, is_web, chunk, keep_system, api_env=None):
    try:
        # if True:
        std_dataset = [json.loads(line)]
        for std_data in std_dataset:
            trajectory = Trajectory(**std_data)
            id = trajectory.id
            events = trajectory.content
            details = trajectory.details

            conversations = []
            previous_web_actions = []

            # Add system message similar to OH Browsing Agent if the dataset is web dataset
            if is_web == "yes":
                # Define action subsets for web datasets (not used directly but kept for documentation)
                action_subsets = ["chat", "bid"]
                if USE_NAV:
                    action_subsets.append("nav")
                # Note: We're not using action_space directly, but this would be the configuration
            for i in range(len(events)):
                event = events[i]
                if (
                    hasattr(event, "source") and event.source == "system"
                ):  # Ignore dataset specific system messages since we have a unified system prompt
                    continue
                try:
                    message = standardized_event_to_openhands_message(
                        id, event, details, previous_web_actions, is_web, chunk, api_env
                    )
                    # prepend original system message to first user message if want to keep original system message from std
                    if (
                        keep_system == "yes"
                        and i == 1
                        and hasattr(events[0], "source")
                        and events[0].source == "system"
                    ):
                        message["value"] = events[0].content + "\n\n" + message["value"]
                    if len(conversations) == 0:
                        # append api function docs to first user message when available
                        if api_env:
                            message["value"] += "\n\n" + get_api_tool_description(dataset, api_env)
                        conversations.extend([message])
                        continue
                    # code to process multiple consecutive function calls + observations
                    elif (
                        message["from"] == "function_call"
                        and conversations[-1]["from"] == "function_call"
                    ):
                        conversations[-1]["value"] = (
                            conversations[-1]["value"]
                            + "\n"
                            + message["value"].replace("THOUGHT: ", "")
                        )
                        # Check if function_call key exists
                        if "function_call" not in conversations[-1]:
                            # First function call, just add the message value
                            continue
                        # if the previous event contains only one function call
                        elif isinstance(conversations[-1]["function_call"], str):
                            conversations[-1]["function_call"] = [
                                conversations[-1]["function_call"],
                                message["function_call"],
                            ]
                        # if the previous event already contains multiple function calls
                        elif isinstance(conversations[-1]["function_call"], list):
                            conversations[-1]["function_call"].append(message["function_call"])
                        else:
                            raise ValueError(
                                f"Unknown function_call type: {type(conversations[-1]['function_call'])}\n{conversations[-1]['function_call']}"
                            )
                        continue
                    if conversations[-1]["from"] == "function_call" and isinstance(
                        event, TextObservation
                    ):
                        message["from"] = "observation"
                        function_name = extract_function_call(conversations[-1]["value"])
                        if function_name:
                            message["value"] = (
                                f"EXECUTION RESULT of [{function_name}]:\n" + message["value"]
                            )
                    conversations.extend([message])
                except Exception as e:
                    traceback.print_exc()
                    print(e)
                    return None

            return {
                "id": trajectory.id,
                "conversations": conversations,
                "system": get_system_message(),
            }
    except Exception as e:
        traceback.print_exc()
        print(e)
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert standardized data to SFT format")
    # parser.add_argument('--output_dataset', type=str, help='Output Dataset name', default='sample_sft.json')
    parser.add_argument("--chunk", type=str, help="Dataset name", required=True)
    parser.add_argument(
        "--is_web", type=str, choices=["yes", "no"], help="Is Dataset type web api", required=True
    )
    parser.add_argument(
        "--keep_system",
        type=str,
        choices=["yes", "no"],
        help="Keep system prompt in first user message or not",
        required=True,
    )
    parser.add_argument(
        "--api_env",
        type=str,
        choices=openhands_default_tools + [None],
        help="The environment in which the APIs are pre-defined",
        default=None,
    )
    args = parser.parse_args()

    output_lines = []
    for line in sys.stdin:
        print(f"Processing line: {line[:100]}...", file=sys.stderr)
        output_line = process_row(
            line,
            is_web=args.is_web,
            chunk=args.chunk,
            keep_system=args.keep_system,
            api_env=args.api_env,
        )
        if output_line:
            print("Successfully processed line", file=sys.stderr)
            output_lines.append(output_line)
        else:
            print("Failed to process line", file=sys.stderr)

    # Print the output as a JSON array
    if output_lines:
        print(json.dumps(output_lines, indent=2))


if __name__ == "__main__":
    main()
