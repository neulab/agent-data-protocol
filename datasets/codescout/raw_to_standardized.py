import json
import sys
from collections import OrderedDict
from typing import Any

from schema_raw import Message, SchemaRaw, ToolCall

from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


def normalize_content(content: str | list | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    parts = []
    for block in content:
        if hasattr(block, "text") and block.text is not None:
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("text") is not None:
            parts.append(block["text"])
        else:
            if hasattr(block, "model_dump"):
                block = block.model_dump(exclude_none=True)
            parts.append(json.dumps(block, ensure_ascii=False))
    return "".join(parts)


def parse_arguments(arguments: str | None) -> dict[str, Any]:
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {"raw_arguments": arguments}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def tool_observations(messages: list[Message], start: int) -> tuple[OrderedDict[str, Message], int]:
    observations: OrderedDict[str, Message] = OrderedDict()
    index = start
    while index < len(messages) and messages[index].role == "tool":
        key = messages[index].tool_call_id or f"tool_{index}"
        observations[key] = messages[index]
        index += 1
    return observations, index


def convert_tool_call(tool_call: ToolCall, assistant_text: str) -> CodeAction | MessageAction:
    function_name = tool_call.function.name
    kwargs = parse_arguments(tool_call.function.arguments)
    description = assistant_text.strip() or None

    if function_name == "terminal":
        command = kwargs.get("command")
        if command is None:
            command = json.dumps(kwargs, ensure_ascii=False)
        return CodeAction(language="bash", content=str(command), description=description)

    if function_name == "localization_finish":
        message = json.dumps(kwargs.get("locations", kwargs), ensure_ascii=False, indent=2)
        return MessageAction(content=f"<finish> {message} </finish>", description=description)

    message = json.dumps({"function": function_name, "arguments": kwargs}, ensure_ascii=False)
    return MessageAction(content=message, description=description)


def trajectory_id(data: SchemaRaw) -> str:
    source_config = data.source_config or "default"
    if data.instance_id:
        parts = [source_config, data.instance_id]
        if data.step is not None:
            parts.append(f"step{data.step}")
        if data.rollout_number is not None:
            parts.append(f"rollout{data.rollout_number}")
        return "codescout_" + "_".join(str(part) for part in parts)
    return f"codescout_{source_config}_{data.source_split}_{data.row_id}"


def process_data(data: SchemaRaw) -> Trajectory | None:
    messages = data.chat_messages.messages if data.chat_messages else data.messages
    if not messages:
        return None

    content = []
    index = 0
    while index < len(messages):
        message = messages[index]
        text = normalize_content(message.content)

        if message.role == "system":
            index += 1
            continue

        if message.role == "user":
            content.append(TextObservation(content=text, source="user"))
            index += 1
            continue

        if message.role == "tool":
            content.append(TextObservation(content=text, source="environment"))
            index += 1
            continue

        if message.role == "assistant":
            if message.tool_calls:
                observations, next_index = tool_observations(messages, index + 1)
                used_observations = set()
                for tool_call in message.tool_calls:
                    content.append(convert_tool_call(tool_call, text))
                    observation = observations.get(tool_call.id or "")
                    if observation is not None:
                        used_observations.add(tool_call.id)
                        content.append(
                            TextObservation(
                                content=normalize_content(observation.content),
                                source="environment",
                            )
                        )
                for tool_id, observation in observations.items():
                    if tool_id not in used_observations:
                        content.append(
                            TextObservation(
                                content=normalize_content(observation.content),
                                source="environment",
                            )
                        )
                index = next_index
            elif text.strip():
                content.append(MessageAction(content=text))
                index += 1
            else:
                index += 1
            continue

        raise ValueError(f"Unsupported message role: {message.role}")

    if not content:
        return None

    details = {
        "source_dataset": data.source_dataset,
        "source_config": data.source_config or "default",
        "source_split": data.source_split,
        "row_id": str(data.row_id),
    }
    if data.instance_id:
        details["instance_id"] = data.instance_id
    if data.step is not None:
        details["step"] = str(data.step)
    if data.rollout_number is not None:
        details["rollout_number"] = str(data.rollout_number)
    if data.reward_dict:
        details["reward_dict"] = data.reward_dict.model_dump_json(exclude_none=True)
    if data.chat_messages and data.chat_messages.tools:
        details["tools"] = json.dumps(data.chat_messages.tools, ensure_ascii=False)

    return Trajectory(id=trajectory_id(data), content=content, details=details)


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        if standardized_data:
            print(standardized_data.model_dump_json())
