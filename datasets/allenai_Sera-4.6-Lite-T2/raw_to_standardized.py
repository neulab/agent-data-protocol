import json
import sys
from typing import Any

from schema_raw import Message, SchemaRaw

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part.text))
        return "\n".join(part for part in parts if part)
    return str(content)


def strip_observation_prefix(text: str) -> str:
    if text.startswith("OBSERVATION:\n"):
        return text[len("OBSERVATION:\n") :]
    return text


def parse_messages(data: SchemaRaw) -> list[Message]:
    messages = json.loads(data.messages) if isinstance(data.messages, str) else data.messages
    return [message if isinstance(message, Message) else Message(**message) for message in messages]


def tool_arguments(raw_arguments: str) -> dict:
    if not raw_arguments:
        return {}
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {"raw_arguments": raw_arguments}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def convert_assistant_tool_call(function_name: str, kwargs: dict, description: str):
    if function_name == "bash":
        return CodeAction(
            language="bash",
            content=kwargs.get("command", ""),
            description=description,
        )
    if function_name == "submit":
        return CodeAction(
            language="bash",
            content="submit",
            description=description,
        )
    return ApiAction(
        function=function_name,
        kwargs=kwargs,
        description=description,
    )


def process_data(data: SchemaRaw) -> Trajectory:
    content = []
    system_prompt = ""

    for message in parse_messages(data):
        role = message.role
        message_text = content_to_text(message.content)

        if role == "system":
            system_prompt = message_text
            continue
        if role == "user":
            content.append(TextObservation(content=message_text, source="user"))
            continue
        if role == "tool":
            content.append(
                TextObservation(
                    content=strip_observation_prefix(message_text),
                    source="environment",
                )
            )
            continue
        if role != "assistant":
            print(f"Unknown role: {role}", file=sys.stderr)
            continue

        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.type != "function":
                    print(f"Unknown tool call type: {tool_call.type}", file=sys.stderr)
                    continue
                content.append(
                    convert_assistant_tool_call(
                        tool_call.function.name,
                        tool_arguments(tool_call.function.arguments),
                        message_text,
                    )
                )
        elif message_text:
            content.append(MessageAction(content=message_text))

    return Trajectory(
        id=data.instance_id,
        content=content,
        details={
            "source": "allenai/Sera-4.6-Lite-T2",
            "func_name": data.func_name,
            "func_path": data.func_path,
            "docker_image": data.docker_image,
            "problem_statement": data.problem_statement,
            "rollout_patch": data.rollout_patch,
            "target_patch": data.target_patch,
            "system_prompt": system_prompt,
        },
    )


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        print(json.dumps(standardized_data.model_dump(), ensure_ascii=False))
