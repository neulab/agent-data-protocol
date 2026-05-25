import json
import sys
from collections import OrderedDict
from typing import Any

from schema_raw import Message, SchemaRaw, ToolCall

from schema.action.action import Action
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.tool_call_links import create_trajectory_with_tool_call_links
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
        # Use "" as the fallback for missing tool_call_id, matching the
        # `tool_call.id or ""` lookup below so unidentified observations
        # still pair with their producing tool call when there is a single
        # such pair (the only case observed in practice).
        key = messages[index].tool_call_id or ""
        if key in observations:
            # Multiple tool messages sharing the same key — the previous
            # observation will be lost and the new one will be (incorrectly)
            # paired with every same-keyed tool_call. None observed in the
            # CodeScout sources, but surface it loudly if it ever appears
            # when extracting the full 58.9K dataset.
            print(
                f"Warning: duplicate tool_call_id {key!r} at message index {index}",
                file=sys.stderr,
            )
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
            # Defensive fallback for orphaned tool messages — a `tool` message
            # that is NOT immediately preceded by an `assistant` with
            # `tool_calls`. Well-formed raw rollouts pre-consume tool messages
            # in the assistant branch below via `tool_observations()` and
            # advance past them with `index = next_index`, so this branch only
            # fires for malformed inputs (none observed in either CodeScout
            # source). Emit them as environment observations to preserve
            # round-trip fidelity rather than silently dropping the content.
            content.append(TextObservation(content=text, source="environment"))
            index += 1
            continue

        if message.role == "assistant":
            if message.tool_calls:
                observations, next_index = tool_observations(messages, index + 1)
                used_observations = set()
                for tool_call in message.tool_calls:
                    is_finish = tool_call.function.name == "localization_finish"
                    content.append(convert_tool_call(tool_call, text))
                    lookup_key = tool_call.id or ""
                    observation = observations.get(lookup_key)
                    if observation is not None:
                        used_observations.add(lookup_key)
                        # The raw rollouts include a tool-response echo after
                        # `localization_finish` that just repeats the submitted
                        # locations. Real OpenHands sessions terminate immediately
                        # on finish, so we drop the echo to match that pattern;
                        # the locations are already captured in the finish action.
                        if not is_finish:
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

    # CodeScout's per-trajectory reward signal is the multi-level localization F1
    # from the upstream `reward_dict`. Attach it to the agent's final action (the
    # `localization_finish` MessageAction). Some raw rollouts include a trailing
    # tool-response observation that echoes the locations; the reward belongs on
    # the agent's action that produced it, not on that echo observation.
    if (
        data.reward_dict is not None
        and data.reward_dict.multilevel_localization_f1_reward is not None
    ):
        for item in reversed(content):
            if isinstance(item, Action):
                item.reward = data.reward_dict.multilevel_localization_f1_reward
                break

    details = {
        "source_dataset": data.source_dataset,
        "source_config": data.source_config or "default",
        "source_split": data.source_split,
        "row_id": str(data.row_id),
    }
    if data.instance_id:
        details["instance_id"] = data.instance_id
    if data.step is not None:
        details["step"] = data.step
    if data.rollout_number is not None:
        details["rollout_number"] = data.rollout_number

    return create_trajectory_with_tool_call_links(id=trajectory_id(data), content=content, details=details)


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        if standardized_data:
            print(standardized_data.model_dump_json())
