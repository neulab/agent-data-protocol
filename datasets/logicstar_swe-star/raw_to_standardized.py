import json
import re
import sys
from typing import Any

from schema_raw import SchemaRaw

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

FUNCTION_RE = re.compile(r"<function=([^>\n]+)>\s*(.*?)\s*</function>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)
EXECUTION_RESULT_RE = re.compile(r"^EXECUTION RESULT of \[[^\]]+\]:\n?")


def parse_parameter_value(value: str) -> Any:
    value = value.strip("\n")
    stripped = value.strip()
    if stripped.startswith(("[", "{")) and stripped.endswith(("]", "}")):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    if re.fullmatch(r"-?\d+", stripped):
        return int(stripped)
    if stripped == "true":
        return True
    if stripped == "false":
        return False
    return value


def parse_parameters(function_body: str) -> dict[str, Any]:
    return {name: parse_parameter_value(value) for name, value in PARAM_RE.findall(function_body)}


def normalize_observation(content: str) -> str:
    return EXECUTION_RESULT_RE.sub("", content, count=1)


def convert_function_call(
    function_name: str, kwargs: dict[str, Any], description: str | None
) -> CodeAction | ApiAction | MessageAction:
    if function_name in {"execute_bash", "bash"}:
        return CodeAction(
            language="bash",
            content=str(kwargs.get("command", "")),
            description=description,
        )
    if function_name in {"finish", "submit"}:
        message = str(kwargs.get("message", "Task submitted."))
        return MessageAction(
            content=f"<finish> {message.strip()} </finish>",
            description=description,
        )
    return ApiAction(function=function_name, kwargs=kwargs, description=description)


def convert_assistant_message(content: str) -> list[CodeAction | ApiAction | MessageAction]:
    actions = []
    current_pos = 0

    for match in FUNCTION_RE.finditer(content):
        thought = content[current_pos : match.start()].strip() or None
        function_name = match.group(1).strip()
        kwargs = parse_parameters(match.group(2))
        actions.append(convert_function_call(function_name, kwargs, thought))
        current_pos = match.end()

    remaining = content[current_pos:].strip()
    if remaining:
        actions.append(MessageAction(content=remaining))
    if not actions:
        actions.append(MessageAction(content=content))
    return actions


def process_data(data: SchemaRaw) -> Trajectory | None:
    messages = json.loads(data.stitched)
    content = []
    seen_user_task = False

    for message in messages:
        role = message["role"]
        message_content = message.get("content") or ""
        if role == "system":
            continue
        if role == "user":
            if not seen_user_task:
                content.append(TextObservation(content=message_content, source="user"))
                seen_user_task = True
            elif message_content.startswith("Submitted:"):
                continue
            else:
                content.append(
                    TextObservation(
                        content=normalize_observation(message_content),
                        source="environment",
                    )
                )
        elif role == "assistant":
            content.extend(convert_assistant_message(message_content))
        else:
            print(f"Unknown role: {role}", file=sys.stderr)

    if not content:
        return None

    has_finish = any(
        isinstance(event, MessageAction) and "<finish>" in event.content for event in content
    )
    if data.resolved and not has_finish:
        content.append(
            TextObservation(content="Task completed successfully.", source="environment")
        )
        content.append(MessageAction(content="<finish> Task completed successfully. </finish>"))

    return Trajectory(
        id=data.instance_id,
        content=content,
        details={
            "timestamp": str(data.timestamp),
            "exit_status": data.exit_status,
            "resolved": str(data.resolved),
            "generated_patch": data.result,
            "source_field": "stitched",
        },
    )


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        if not data.resolved:
            continue
        standardized_data = process_data(data)
        if standardized_data:
            print(standardized_data.model_dump_json())
