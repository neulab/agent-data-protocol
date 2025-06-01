import inspect
import json
import re
import shlex
import sys

import api
from schema_raw import SchemaRaw

from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

ACTIONS = [f[0] for f in inspect.getmembers(api, inspect.isfunction) if f[0] != "run"]


def parse_edit_action(action_str):
    first_arg = action_str.split()[1]
    if re.match(r"^\d+:\d+$", first_arg):
        start_line, end_line = map(int, first_arg.split(":"))
        replacement_text = action_str.split(None, 2)[-1].strip()
    else:
        # sometimes agent outputs "edit filename.py 1:10 replacement_text" instead of "edit 1:10 replacement_text"
        start_line, end_line = map(int, action_str.split()[2].split(":"))
        replacement_text = action_str.split(None, 3)[-1].strip()
    if replacement_text.endswith("end_of_edit"):
        replacement_text = replacement_text[: -len("end_of_edit")]
    return {"start_line": start_line, "end_line": end_line, "replacement_text": replacement_text}


def parse_api_action(item):
    thought, action_str, remainder = item.text.rsplit("```", 2)
    thought, remainder = thought.strip(), remainder.strip()
    thought = thought[len("DISCUSSION") :].strip() if thought.startswith("DISCUSSION") else thought
    thought = thought[: -len("COMMAND")].strip() if thought.endswith("COMMAND") else thought
    thought = thought + " " + remainder if remainder else thought
    codeblock_lang = re.fullmatch(r"\w+\s*", action_str.splitlines()[0])
    if codeblock_lang:
        action_str = "\n".join(action_str.splitlines()[1:])
    action_str = action_str.strip()
    action_name = action_str.split()[0]
    if action_name in ACTIONS:
        if action_name == "edit":
            action_kwargs = parse_edit_action(action_str)
        else:
            action_args = shlex.split(action_str)[1:]
            action_params = inspect.signature(getattr(api, action_name)).parameters
            action_kwargs = {param: arg for param, arg in zip(action_params.keys(), action_args)}
        return ApiAction(function=action_name, kwargs=action_kwargs, description=thought)
    else:
        return ApiAction(function="run", kwargs={"command": action_str}, description=thought)


def process_item(item):
    if item.role == "system":
        return (
            TextObservation(content=item.system_prompt, source="system")
            if item.system_prompt
            else None
        )
    elif item.role == "user":
        return TextObservation(content=item.text, source="user")
    elif item.role == "ai" and "```" in item.text:
        try:
            return parse_api_action(item)
        except Exception:
            return MessageAction(content=item.text)
    elif item.role == "ai":
        return MessageAction(content=item.text)
    else:
        print(f"Unknown role: {item.role}", file=sys.stderr)
        return None


def process_data(data):
    content = []
    for item in data.trajectory:
        observation = process_item(item)
        if observation is not None:
            content.append(observation)

    return Trajectory(
        id=data.instance_id,
        content=content,
        details={
            "model_name": data.model_name or "",
            "exit_status": data.exit_status or "",
            "generated_patch": data.generated_patch or "",
            "eval_logs": data.eval_logs or "",
        },
    )


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        print(standardized_data.model_dump_json())
