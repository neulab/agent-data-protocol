import json
import sys
from typing import Any

from schema_raw import SchemaRaw

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

PRIOR_INS = "2. Create a script to reproduce the error and execute it with `python <filename.py>` using the bash tool, to confirm the error"
NEW_INS = "2. Create a script to reproduce the error and execute it with `python <filename.py>`, to confirm the error"


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "".join(parts)
    return str(content)


def clean_observation(text: str) -> str:
    if text.startswith("OBSERVATION:\n"):
        return text[len("OBSERVATION:\n") :]
    return text


def convert_tool_call(tool_call: dict, description: str | None):
    if tool_call.get("type") != "function":
        print(f"Unknown tool call type: {tool_call.get('type')}", file=sys.stderr)
        return None

    function = tool_call.get("function", {})
    name = function.get("name")
    try:
        kwargs = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError:
        print(f"Failed to parse tool arguments for {name}: {function.get('arguments')}", file=sys.stderr)
        kwargs = {}

    if name == "bash":
        return CodeAction(
            language="bash",
            content=kwargs.get("command", ""),
            description=description,
        )
    if name in {"str_replace_editor", "submit"}:
        return ApiAction(function=name, kwargs=kwargs, description=description)

    print(f"Unknown tool function: {name}", file=sys.stderr)
    return ApiAction(function=name, kwargs=kwargs, description=description)


def convert_message(message: dict, idx: int) -> list:
    role = message.get("role")
    text = content_to_text(message.get("content"))

    if role == "system":
        return []
    if role == "user":
        if idx == 1:
            text = text.replace(PRIOR_INS, NEW_INS)
        return [TextObservation(content=text, source="user")]
    if role == "tool":
        return [TextObservation(content=clean_observation(text), source="environment")]
    if role == "assistant":
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return [MessageAction(content=text)] if text else []
        actions = []
        for tool_call in tool_calls:
            action = convert_tool_call(tool_call, text or None)
            if action is not None:
                actions.append(action)
        return actions

    print(f"Unknown role: {role}", file=sys.stderr)
    return []


def process_data(data: SchemaRaw) -> Trajectory | None:
    raw_messages = json.loads(data.messages)
    content = []
    for idx, message in enumerate(raw_messages):
        content.extend(convert_message(message, idx))

    if not content:
        return None

    if not isinstance(content[-1], MessageAction) or "<finish>" not in content[-1].content:
        content.append(TextObservation(content="Task completed successfully.", source="user"))
        content.append(
            MessageAction(
                content="<finish> Task completed successfully. </finish>",
                description="",
            )
        )

    return Trajectory(
        id=data.instance_id,
        content=content,
        details={
            "func_name": data.func_name or "",
            "func_path": data.func_path or "",
            "problem_statement": data.problem_statement or "",
            "rollout_patch": data.rollout_patch or "",
            "target_patch": data.target_patch or "",
            "docker_image": data.docker_image or "",
        },
    )


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        if standardized_data:
            print(standardized_data.model_dump_json())
