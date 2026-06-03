import json
import re
import sys
from typing import Any

from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.tool_call_links import create_trajectory_with_tool_call_links
from schema.trajectory import Trajectory

PROMPT_LINE_RE = re.compile(r"(?m)^[^\n]*# .*$")


def extract_json_object(content: str) -> dict[str, Any] | None:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def split_observation_chunks(content: str) -> list[str]:
    content = content.replace("New Terminal Output:\n", "", 1).strip("\n")
    matches = list(PROMPT_LINE_RE.finditer(content))
    if len(matches) <= 1:
        return [content] if content else []

    chunks = []
    header = content[: matches[0].start()]
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        chunk = content[start:end]
        if index == 0 and header:
            chunk = header + chunk
        chunks.append(chunk)
    return chunks


def assistant_actions(content: str) -> list[CodeAction | MessageAction]:
    data = extract_json_object(content)
    if data is None:
        return [MessageAction(content=content.strip(), description=None)]

    description = data.get("analysis") or data.get("plan")
    actions: list[CodeAction | MessageAction] = []
    for index, command in enumerate(data.get("commands") or []):
        if isinstance(command, dict):
            keystrokes = command.get("keystrokes", "")
        elif isinstance(command, str):
            keystrokes = command
        else:
            continue
        keystrokes = keystrokes.rstrip("\n")
        if not keystrokes:
            continue
        actions.append(
            CodeAction(
                language="bash",
                content=keystrokes,
                description=description if index == 0 else None,
            )
        )

    if data.get("task_complete"):
        message = data.get("analysis") or "Task completed successfully."
        actions.append(MessageAction(content=f"<finish> {message} </finish>", description=None))

    if actions:
        return actions

    visible_json = json.dumps(data, ensure_ascii=False, indent=2)
    return [MessageAction(content=visible_json, description=description)]


def process_trajectory(raw_data: dict[str, Any]) -> Trajectory | None:
    content = []
    pending_code = []
    pending_messages = []
    first_human_turn = True

    for turn in raw_data["conversations"]:
        role = turn["from"]
        value = turn["value"]

        if role == "human":
            if first_human_turn:
                content.append(TextObservation(content=value, source="user"))
                first_human_turn = False
                continue

            observations = [
                TextObservation(content=chunk, source="environment")
                for chunk in split_observation_chunks(value)
            ]
            index = 0
            while index < len(observations) and pending_code:
                content.append(pending_code.pop(0))
                content.append(observations[index])
                index += 1
            content.extend(observations[index:])
            if not pending_code and pending_messages:
                content.extend(pending_messages)
                pending_messages = []
            continue

        if role == "gpt":
            for action in assistant_actions(value):
                if isinstance(action, CodeAction):
                    pending_code.append(action)
                elif pending_code:
                    pending_messages.append(action)
                else:
                    content.append(action)

    content.extend(pending_code)
    content.extend(pending_messages)
    if not content:
        return None

    return create_trajectory_with_tool_call_links(
        id=f"litecoder-terminal-sft-{raw_data['id']}",
        content=content,
    )


if __name__ == "__main__":
    for line in sys.stdin:
        trajectory = process_trajectory(json.loads(line))
        if trajectory is not None:
            print(trajectory.model_dump_json())
