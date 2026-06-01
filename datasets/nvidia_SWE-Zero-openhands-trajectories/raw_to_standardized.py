import json
import sys
from typing import Any

from schema_raw import SchemaRaw

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.tool_call_links import create_trajectory_with_tool_call_links


def parse_arguments(arguments: str | dict[str, Any] | None) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    return json.loads(arguments)


def normalize_tool_observation(content: str) -> str:
    if "OBSERVATION:\n" in content:
        return "\n".join(content.split("OBSERVATION:\n")[1:])
    return content


def process_assistant_message(message):
    if not message.tool_calls:
        return [MessageAction(content=message.content or "")]

    actions = []
    for tool_call in message.tool_calls:
        if tool_call.type != "function":
            print(f"Unknown tool call type: {tool_call.type}", file=sys.stderr)
            continue

        function_name = tool_call.function.name
        kwargs = parse_arguments(tool_call.function.arguments)
        thought = message.content or None

        if function_name == "execute_bash":
            actions.append(
                CodeAction(
                    language="bash",
                    content=kwargs.get("command", ""),
                    description=thought,
                )
            )
        elif function_name == "finish":
            message_text = kwargs.get("message") or "Task completed."
            actions.append(
                MessageAction(
                    content=f"<finish> {message_text} </finish>",
                    description=thought,
                )
            )
        else:
            actions.append(
                ApiAction(
                    function=function_name,
                    kwargs=kwargs,
                    description=thought,
                )
            )
    return actions


def process_data(data: SchemaRaw):
    content = []
    for message in data.trajectory:
        if message.role == "system":
            continue
        if message.role == "user":
            content.append(TextObservation(content=message.content or "", source="user"))
        elif message.role == "tool":
            content.append(
                TextObservation(
                    content=normalize_tool_observation(message.content or ""),
                    source="environment",
                )
            )
        elif message.role == "assistant":
            content.extend(process_assistant_message(message))
        else:
            print(f"Unknown role: {message.role}", file=sys.stderr)

    if not content:
        return None

    return create_trajectory_with_tool_call_links(
        id=data.trajectory_id,
        content=content,
        details={
            "instance_id": data.instance_id,
            "repo": data.repo,
            "source_license": data.license,
            "source_dataset": data.dataset,
            "model_patch": data.model_patch,
        },
    )


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        if standardized_data:
            print(standardized_data.model_dump_json())
