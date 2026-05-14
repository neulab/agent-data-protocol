import json
import sys
from typing import Any

from schema_raw import SchemaRaw

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

FINISH_MESSAGE = "<finish> I have successfully completed the task. </finish>"
SUCCESS_OBSERVATION = "Task completed successfully."


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
    if not data.resolved:
        return None

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
                    name=message.name,
                )
            )
        elif message.role == "assistant":
            content.extend(process_assistant_message(message))
        else:
            print(f"Unknown role: {message.role}", file=sys.stderr)

    if not content:
        return None

    if not (
        isinstance(content[-1], MessageAction)
        and "<finish>" in content[-1].content
        and "</finish>" in content[-1].content
    ):
        content.append(TextObservation(content=SUCCESS_OBSERVATION, source="user"))
        content.append(MessageAction(content=FINISH_MESSAGE))

    return Trajectory(
        id=data.trajectory_id,
        content=content,
        details={
            "instance_id": data.instance_id,
            "repo": data.repo,
            "exit_status": data.exit_status or "",
            "resolved": str(data.resolved),
            "gen_tests_correct": "" if data.gen_tests_correct is None else str(data.gen_tests_correct),
            "pred_passes_gen_tests": ""
            if data.pred_passes_gen_tests is None
            else str(data.pred_passes_gen_tests),
            "model_patch": data.model_patch or "",
            "tools": json.dumps(data.tools, indent=2),
        },
    )


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        if standardized_data:
            print(standardized_data.model_dump_json())
