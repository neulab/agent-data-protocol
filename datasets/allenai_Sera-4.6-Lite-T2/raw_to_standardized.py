import json
import re
import sys
from typing import Any

from schema_raw import Message, SchemaRaw

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.tool_call_links import create_trajectory_with_tool_call_links
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


THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def strip_observation_prefix(text: str) -> str:
    if text.startswith("OBSERVATION:\n"):
        return text[len("OBSERVATION:\n") :]
    return text


def split_think_blocks(text: str) -> tuple[str, str | None]:
    think_blocks = [match.strip() for match in THINK_BLOCK_RE.findall(text)]
    visible_text = THINK_BLOCK_RE.sub("", text).strip()
    reasoning_content = "\n\n".join(block for block in think_blocks if block)
    return visible_text, reasoning_content or None


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


def convert_assistant_tool_call(
    function_name: str,
    kwargs: dict,
    description: str,
    reasoning_content: str | None,
):
    if function_name == "bash":
        return CodeAction(
            language="bash",
            content=kwargs.get("command", ""),
            description=description,
            reasoning_content=reasoning_content,
        )
    if function_name == "submit":
        return CodeAction(
            language="bash",
            content="submit",
            description=description,
            reasoning_content=reasoning_content,
        )
    return ApiAction(
        function=function_name,
        kwargs=kwargs,
        description=description,
        reasoning_content=reasoning_content,
    )


def process_data(data: SchemaRaw) -> Trajectory:
    content = []

    for message in parse_messages(data):
        role = message.role
        message_text = content_to_text(message.content)

        if role == "system":
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

        visible_text, reasoning_content = split_think_blocks(message_text)
        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.type != "function":
                    print(f"Unknown tool call type: {tool_call.type}", file=sys.stderr)
                    continue
                content.append(
                    convert_assistant_tool_call(
                        tool_call.function.name,
                        tool_arguments(tool_call.function.arguments),
                        visible_text,
                        reasoning_content,
                    )
                )
        elif visible_text or reasoning_content:
            content.append(
                MessageAction(
                    content=visible_text,
                    reasoning_content=reasoning_content,
                )
            )

    return create_trajectory_with_tool_call_links(
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
        },
    )


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        print(json.dumps(standardized_data.model_dump(), ensure_ascii=False))
