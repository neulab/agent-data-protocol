import json
import sys
from typing import Any

from schema_raw import Message, SchemaRaw, ToolCall

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.tool_call_links import create_trajectory_with_tool_call_links

BASH_TOOLS = {"bash", "execute_bash", "terminal"}
FINISH_TOOLS = {"submit", "finish"}


def parse_arguments(arguments: str | dict[str, Any] | None) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {"command": arguments}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def strip_observation_prefix(content: str) -> str:
    for prefix in ["OBSERVATION:\n", "OBSERVATION:\r\n"]:
        if content.startswith(prefix):
            return content[len(prefix) :]
    return content


def tool_call_to_action(msg: Message, tool_call: ToolCall):
    function_name = tool_call.function.name
    kwargs = parse_arguments(tool_call.function.arguments)
    description = msg.content or None
    reasoning_content = msg.reasoning_content or None

    if function_name in BASH_TOOLS:
        command = kwargs.get("command") or kwargs.get("cmd") or ""
        if not isinstance(command, str):
            command = json.dumps(command, ensure_ascii=False)
        return CodeAction(
            language="bash",
            content=command,
            description=description,
            reasoning_content=reasoning_content,
        )

    if function_name in FINISH_TOOLS:
        message = kwargs.get("message") or msg.content or "Task completed."
        return MessageAction(
            content=f"<finish> {message} </finish>",
            description=description if msg.content and msg.content != message else None,
            reasoning_content=reasoning_content,
        )

    return ApiAction(
        function=function_name,
        kwargs=kwargs,
        description=description,
        reasoning_content=reasoning_content,
    )


def message_to_events(msg: Message):
    role = msg.role
    content = msg.content or ""

    if role == "system":
        return []
    if role == "user":
        return [TextObservation(content=content, source="user")]
    if role == "tool":
        return [TextObservation(content=strip_observation_prefix(content), source="environment")]
    if role != "assistant":
        print(f"Unknown role: {role}", file=sys.stderr)
        return []

    if msg.tool_calls:
        return [tool_call_to_action(msg, tool_call) for tool_call in msg.tool_calls]
    if content or msg.reasoning_content:
        return [
            MessageAction(
                content=content,
                reasoning_content=msg.reasoning_content or None,
            )
        ]
    return []


def add_finish_if_missing(content):
    if not content:
        return content
    last = content[-1]
    if isinstance(last, MessageAction) and "<finish>" in last.content:
        return content
    finish = MessageAction(
        content="<finish> I have completed the task. </finish>",
        description="",
    )
    if isinstance(last, Observation):
        content.append(finish)
    else:
        content.append(TextObservation(content="Task completed successfully.", source="user"))
        content.append(finish)
    return content


def process_data(data: SchemaRaw, row_number: int = 0):
    content = []
    for msg in data.messages:
        content.extend(message_to_events(msg))

    content = add_finish_if_missing(content)
    if not content or not isinstance(content[0], TextObservation):
        return None

    return create_trajectory_with_tool_call_links(
        id=data.sample_name or data.id or f"gair_davinci_dev_{row_number}",
        content=content,
        details={
            "source": "GAIR/daVinci-Dev",
            "config": "env_native",
            "sample_name": data.sample_name or data.id or "",
        },
    )


if __name__ == "__main__":
    for row_number, line in enumerate(sys.stdin):
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data, row_number)
        if standardized_data is not None:
            print(json.dumps(standardized_data.model_dump(), ensure_ascii=False))
