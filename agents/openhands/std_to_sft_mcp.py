import argparse
import json
import os
import random
import re
import sys
import traceback

from agents.openhands.api import get_api_tool_description
from agents.openhands.convert_api_to_mcp import (
    get_api_tools,
    get_language_tools,
)
from agents.openhands.system_prompt.system import get_system_message
from agents.openhands.system_prompt.user import get_web_user_message
from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory
from scripts.html_to_axtree import HTMLToAXTree

dataset = os.getenv("MY_DATASET")
assert dataset, "Please set the environment variable MY_DATASET"

openhands_default_tools = {
    "execute_bash": {"required": ["command"], "optional": ["is_input"]},
    "think": {"required": ["thought"], "optional": []},
    "finish": {"required": ["message", "task_completed"], "optional": []},
    "web_read": {"required": ["url"], "optional": []},
    "browser": {"required": ["code"], "optional": []},
    "execute_ipython_cell": {"code": ["command"], "optional": []},
    "str_replace_editor": {
        "required": ["command", "path"],
        "optional": ["file_text", "old_str", "new_str", "insert_line", "view_range"],
    },
    "edit_file": {"required": ["path", "content"], "optional": ["start", "end"]},
}

action_function = {"python": "execute_ipython_cell", "bash": "execute_bash", "web": "browser"}

function_args = {"execute_ipython_cell": "code", "execute_bash": "command", "browser": "code"}

browser_default_apis = {
    "goto": {"required": ["url"], "optional": []},
    "go_back": {"required": [], "optional": []},
    "go_forward": {"required": [], "optional": []},
    "noop": {"required": [], "optional": ["wait_ms"]},
    "scroll": {"required": ["delta_x", "delta_y"], "optional": []},
    "fill": {"required": ["bid", "value"], "optional": []},
    "select_option": {"required": ["bid", "options"], "optional": []},
    "click": {"required": ["bid"], "optional": ["button", "modifiers"]},
    "dblclick": {"required": ["bid"], "optional": ["button", "modifiers"]},
    "hover": {"required": ["bid"], "optional": []},
    "press": {"required": ["bid", "key_comb"], "optional": []},
    "focus": {"required": ["bid"], "optional": []},
    "clear": {"required": ["bid"], "optional": []},
    "drag_and_drop": {"required": ["from_bid", "to_bid"], "optional": []},
    "upload_file": {"required": ["bid", "file"], "optional": []},
}

USE_NAV = (
    os.environ.get("USE_NAV", "true") == "true"
)  # only disable NAV actions when running webarena and miniwob benchmarks

generate_axtree = HTMLToAXTree(dataset)


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


# Extract the tool in a OH format function call
def extract_function_call(content):
    for tool in openhands_default_tools:
        if f"<function={tool}" in content:
            return tool
    return None


NON_OH_EVENTS = {}
PREV_BID = None


def standardized_event_to_openhands_message(
    id,
    event: ApiAction | CodeAction | MessageAction | TextObservation | WebObservation,
    previous_web_actions: list,
    is_web: bool,
    chunk: str,
    api_env: str = None,
    api_sigs=None,
    languages: list = [],
    mcp_tools: dict = {},
) -> dict:
    global PREV_BID
    if isinstance(event, WebObservation):
        if event.axtree is not None:
            axtree = event.axtree
        elif generate_axtree.last_html != event.html:
            axtree = generate_axtree.build_axtree(id, event.html, chunk)
        else:
            axtree = generate_axtree.last_xtree
        prompt = get_web_user_message("", event.url, axtree, PREV_BID)
        return {"from": "human", "value": prompt}

    if isinstance(event, ApiAction):
        PREV_BID = None
        thought = event.description + "\n\n" if event.description else ""
        function_name = event.function
        arguments = {k: v for k, v in event.kwargs.items() if k not in ["element_id", "xpath"]}

        # for tool that are one of the default OH tools
        if function_name in openhands_default_tools and function_name not in api_sigs:
            tool_args = openhands_default_tools[function_name]
            if not verify_args(tool_args["required"], tool_args["optional"], arguments):
                raise ValueError(f"Function call with wrong argument: {event}")
            function_call = format_function(function_name, arguments)
            return {"from": "function_call", "value": f"{thought}{function_call}"}

        # for OH default browser based apis that don't require bid
        if (
            is_web
            and function_name in browser_default_apis
            and function_name not in api_sigs
            and "bid" not in browser_default_apis[function_name]["required"]
        ):
            api_args = browser_default_apis[function_name]
            if not verify_args(api_args["required"], api_args["optional"], arguments):
                raise ValueError(f"Function call with wrong argument: {event}")
            api_action = f"{function_name}({', '.join([f'{k}={arguments[k]}' for k in arguments])})"
            previous_web_actions.extend([api_action])
            function_call = format_function("browser", {"code": api_action})
            return {"from": "function_call", "value": f"{thought}{function_call}"}

        # try to directly get the browsergym_id from the event kwargs
        browsergym_id = event.kwargs.get("bid", None)
        if not browsergym_id:
            browsergym_id = event.kwargs.get("element_id", None)
        # this gets the browsergym_id of the element that the user is interacting with
        # the latest(last seen) html's obs is updated whenever build_axtree is called
        # the latest obs is used to get the browsergym_id
        if not browsergym_id:
            event_xpath = event.kwargs.get("xpath", None)
            if event_xpath:
                browsergym_id = generate_axtree.get_bid(id, event_xpath, chunk)
        # for tool calls that are not browser based since there is no browsergym_id
        # and tool calls that are specified as non-web
        # these should all be dataset specific apis

        # tools in MCP
        if (not browsergym_id or not is_web) and function_name in mcp_tools:
            assert function_name in api_sigs
            if not verify_args(
                api_sigs[function_name]["required"], api_sigs[function_name]["optional"], arguments
            ):
                raise ValueError(f"Function call with wrong argument: {event}")
            # api_action = f"{function_name}({', '.join([f'{k}={arguments[k]}' for k in arguments])})"
            function_call = format_function(function_name, {k: arguments[k] for k in arguments})
            return {"from": "function_call", "value": f"{thought}{function_call}"}

        # tools in normal api action
        if (not browsergym_id or not is_web) and function_name in api_sigs:
            if not api_env:
                # Default to 'execute_ipython_cell' if api_env is not specified
                api_env = "execute_ipython_cell"
            if not verify_args(
                api_sigs[function_name]["required"], api_sigs[function_name]["optional"], arguments
            ):
                raise ValueError(f"Function call with wrong argument: {event}")
            api_action = f"{function_name}({', '.join([f'{k}={arguments[k]}' for k in arguments])})"
            function_call = format_function(
                api_env, {function_args.get(api_env, "code"): api_action}
            )
            return {"from": "function_call", "value": f"{thought}{function_call}"}

        api_env = "browser"
        if not browsergym_id[0] == browsergym_id[-1] == '"':
            browsergym_id = f'"{browsergym_id[0]}"'
        PREV_BID = browsergym_id
        # for apis that are browser based but are not OH default browser apis
        # these should all be dataset specific apis
        if function_name in api_sigs:
            if "bid" in api_sigs[function_name]["required"] and browsergym_id:
                arguments["bid"] = browsergym_id
            if not verify_args(
                api_sigs[function_name]["required"], api_sigs[function_name]["optional"], arguments
            ):
                raise ValueError(f"Function call with wrong argument: {event}")
            api_action = f"{function_name}({', '.join([f'{k}={arguments[k]}' for k in arguments])})"
            function_call = format_function(
                api_env, {function_args.get(api_env, "code"): api_action}
            )
            return {"from": "function_call", "value": f"{thought}{function_call}"}

        # for tool calls that are browser based and need bid
        api_args = browser_default_apis[function_name]
        if browsergym_id:
            arguments["bid"] = browsergym_id

        # to handle mismatching "bid" and "id" arguments
        if "bid" not in arguments:
            if "id" in arguments:
                arguments["bid"] = arguments.pop("id")
                PREV_BID = arguments["bid"]
        if not verify_args(api_args["required"], api_args["optional"], arguments):
            raise ValueError(f"Function call with wrong argument: {event}")
        api_action = f"{function_name}({', '.join([f'{k}={arguments[k]}' for k in arguments])})"
        previous_web_actions.extend([api_action])
        function_call = format_function(api_env, {function_args.get(api_env, "code"): api_action})
        return {"from": "function_call", "value": f"{thought}{function_call}"}

    if isinstance(event, CodeAction):
        thought = event.description + "\n\n" if event.description else ""
        function_name = action_function.get(event.language, f"execute_{event.language}")
        code_content = event.content
        if function_name not in openhands_default_tools:
            if function_name not in NON_OH_EVENTS:
                NON_OH_EVENTS[function_name] = 0
            NON_OH_EVENTS[function_name] += 1
            # raise ValueError(f"Event with unknown code action type: {type(event)}\n{function_name}{event}")
            # return None
            languages.append(event.language)
        arg = function_args.get(function_name, "code")
        code_action = format_function(function_name, {arg: code_content})
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
        if event.source == "user":
            event.source = "human"

        elif event.source == "agent":
            event.source = "gpt"

        elif event.source == "environment":
            event.source = "observation"

        else:
            raise ValueError(f"Wrong event source: {event.source}")
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


def process_row(line, is_web, chunk, api_env, api_tool_description, api_sigs, api_tools):
    std_dataset = [json.loads(line)]
    std_data = std_dataset[0]
    trajectory = Trajectory(**std_data)
    id = trajectory.id
    events = trajectory.content
    # details = trajectory.details
    conversations = []
    previous_web_actions = []
    languages = []
    if is_web or len(api_tools) > 12 or random.choice([True, False]):
        mcp_tools = {}
    else:
        mcp_tools = api_tools
        api_tool_description = ""
    for i in range(len(events)):
        event = events[i]
        try:
            message = standardized_event_to_openhands_message(
                id,
                event,
                previous_web_actions,
                is_web,
                chunk,
                api_env,
                api_sigs,
                languages,
                mcp_tools,
            )
            if not message:
                return None
            if len(conversations) == 0:
                # append api function docs to first user message when available
                if api_env:
                    message["value"] = api_tool_description + message["value"]
                conversations.extend([message])
                continue

            # Combine consecutive user message and web observation
            if conversations[-1]["from"] == "human" and isinstance(event, WebObservation):
                conversations[-1]["value"] += "\n\n" + message["value"]
                continue

            # Match observations to function_calls
            if conversations[-1]["from"] == "function_call" and isinstance(event, TextObservation):
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

    # Handle non python / bash / browser based coding languages
    if languages:
        language_tools = get_language_tools(languages)
    else:
        language_tools = {}
    return {
        "id": trajectory.id,
        "conversations": conversations,
        "system": get_system_message(
            additional_tools=[tool for tool in mcp_tools.values()]
            + [tool for tool in language_tools.values()]
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert standardized data to SFT format")
    parser.add_argument(
        "--is_web",
        type=str,
        choices=["yes", "no"],
        help="Does the dataset contain web api",
        required=True,
        default="no",
    )
    parser.add_argument("--chunk", type=str, help="Dataset name", required=True)
    parser.add_argument(
        "--api_env",
        type=str,
        choices=list(openhands_default_tools.keys()) + [None],
        help="The environment in which the APIs are pre-defined",
        default=None,
    )
    args = parser.parse_args()
    args.is_web = args.is_web == "yes"
    exclude_apis = browser_default_apis if args.is_web else {}
    api_tools = get_api_tools(dataset)
    api_tools = {
        name: tool for name, tool in api_tools.items() if name not in openhands_default_tools
    }
    api_tool_description, api_sigs = get_api_tool_description(dataset, exclude_apis, args.api_env)
    count = 0
    from datetime import datetime

    now = datetime.now()
    print(now, file=sys.stderr)
    for line in sys.stdin:
        if count % 1000 == 0 and count != 0:
            now = datetime.now()
            print(f"Processed {count} lines; {now}", file=sys.stderr)
        output_line = process_row(
            line,
            is_web=args.is_web,
            chunk=args.chunk,
            api_env=args.api_env,
            api_tool_description=api_tool_description,
            api_sigs=api_sigs,
            api_tools=api_tools,
        )
        if output_line:
            # print("Successfully processed line", file=sys.stderr)
            with open(f"datasets/{dataset}/full_sft_mcp.jsonl", "a") as f:
                try:
                    f.write(json.dumps(output_line) + "\n")
                    count += 1
                except Exception as e:
                    traceback.print_exc()
                    print(e)
                    continue
        # else:
        #     print(f"Failed to process line: {line[:10]}...", file=sys.stderr)
    print(f"Number of non OH events: {NON_OH_EVENTS}", file=sys.stderr)
    # if args.is_web:
    #     print(f"Trimming web observation", file=sys.stderr)
    #     parse_sft(f"datasets/{dataset}/full_sft.jsonl", f"datasets/{dataset}/full_sft.jsonl")


if __name__ == "__main__":
    main()
