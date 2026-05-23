import json
import re
import sys
from typing import Any

from schema_raw import ConversationTurn, SchemaRaw

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

SHELL_TOOL_NAMES = {"bash", "run_command", "shell", "terminal", "execute_bash"}
READ_TOOL_NAMES = {"read", "read_file"}
WRITE_TOOL_NAMES = {"write", "write_file"}
EDIT_TOOL_NAMES = {"edit", "edit_file"}
THINK_TOOL_NAMES = {"think", "thinking"}


def parse_json_maybe(value: Any) -> Any:
    if value in (None, ""):
        return {}
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def tool_input(turn: ConversationTurn) -> dict[str, Any]:
    parsed = parse_json_maybe(turn.tool_input_json)
    if isinstance(parsed, dict):
        return parsed
    parsed_content = parse_json_maybe(turn.content)
    if isinstance(parsed_content, dict):
        return parsed_content
    if parsed not in ({}, None):
        return {"input": parsed}
    return {}


def clean_function_name(name: str) -> str:
    cleaned = re.sub(r"\W+", "_", name).strip("_")
    if not cleaned or cleaned[0].isdigit():
        return "generic_tool"
    return cleaned


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def get_path(turn: ConversationTurn, params: dict[str, Any]) -> str | None:
    for key in ("file_path", "notebook_path", "path", "filename", "absolute_path"):
        value = params.get(key)
        if value:
            return str(value)
    return optional_str(turn.file_path)


def as_str_replace_editor(
    turn: ConversationTurn, name: str, params: dict[str, Any]
) -> ApiAction | None:
    path = get_path(turn, params)
    if not path:
        return None

    if name in READ_TOOL_NAMES:
        return ApiAction(function="str_replace_editor", kwargs={"command": "view", "path": path})

    if name in WRITE_TOOL_NAMES:
        file_text = params.get("content", params.get("file_text", params.get("text", "")))
        return ApiAction(
            function="str_replace_editor",
            kwargs={"command": "create", "path": path, "file_text": str(file_text)},
        )

    if name in EDIT_TOOL_NAMES:
        old_str = params.get("old_string", params.get("old_str"))
        new_str = params.get("new_string", params.get("new_str", params.get("replacement")))
        if old_str is None or new_str is None:
            return None
        return ApiAction(
            function="str_replace_editor",
            kwargs={
                "command": "str_replace",
                "path": path,
                "old_str": str(old_str),
                "new_str": str(new_str),
            },
        )

    return None


def convert_tool_use(turn: ConversationTurn) -> CodeAction | ApiAction | None:
    name = (turn.tool_name or "generic_tool").strip() or "generic_tool"
    lower_name = name.lower()
    params = tool_input(turn)

    if lower_name in SHELL_TOOL_NAMES:
        command = params.get("command") or turn.command or turn.content or ""
        return CodeAction(language="bash", content=str(command), description=name)

    if lower_name in THINK_TOOL_NAMES:
        thought = params.get("thought") or turn.content or ""
        return ApiAction(function="think", kwargs={"thought": str(thought)}, description=name)

    editor_action = as_str_replace_editor(turn, lower_name, params)
    if editor_action:
        editor_action.description = name
        return editor_action

    kwargs: dict[str, Any] = {"tool_name": name, "tool_input": params}
    if not params:
        kwargs["content"] = turn.content
    return ApiAction(
        function="generic_tool",
        kwargs=kwargs,
        description=clean_function_name(name),
    )


def convert_turn(turn: ConversationTurn, seen_user_prompt: bool) -> list[Any]:
    content = turn.content or ""
    turn_type = (turn.turn_type or "").lower()
    role = turn.role.lower()

    if turn_type == "user_prompt" or (role == "user" and turn.is_conversational):
        if not content:
            return []
        return [TextObservation(content=content, source="user")]

    if turn_type == "assistant_response" or (role == "assistant" and turn.is_conversational):
        if not content:
            return []
        return [MessageAction(content=content)]

    if turn_type == "assistant_thinking":
        if not content:
            return []
        return [ApiAction(function="think", kwargs={"thought": content})]

    if turn_type == "tool_use" or role == "tool_use":
        action = convert_tool_use(turn)
        return [action] if action else []

    if turn_type == "tool_result" or role == "tool_result":
        if not content:
            return []
        return [TextObservation(content=content, source="environment")]

    if not seen_user_prompt:
        return []

    if content:
        label = turn_type or role or "metadata"
        return [TextObservation(content=f"[{label}]\n{content}", source="environment")]
    return []


def details_from_raw(data: SchemaRaw) -> dict[str, str]:
    details = {"source": "SALT-NLP/SWE-chat", "source_config": "conversations"}
    for key in (
        "repo_id",
        "checkpoint_pk",
        "user_id",
        "agent",
        "strategy",
        "branch",
        "created_at",
        "transcript_path",
        "tool_call_count",
        "turn_count",
        "prompt_count",
        "agent_percentage",
        "user_persona",
        "session_success",
    ):
        value = getattr(data, key)
        if value is not None:
            details[key] = str(value)
    return details


def process_data(data: SchemaRaw) -> Trajectory | None:
    content = []
    seen_user_prompt = False

    for turn in sorted(
        data.turns,
        key=lambda item: item.turn_number if item.turn_number is not None else float("inf"),
    ):
        events = convert_turn(turn, seen_user_prompt)
        if events and any(
            isinstance(event, TextObservation) and event.source == "user" for event in events
        ):
            seen_user_prompt = True
        content.extend(events)

    if not content or not seen_user_prompt:
        return None

    return Trajectory(id=data.session_id, content=content, details=details_from_raw(data))


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        if standardized_data:
            print(standardized_data.model_dump_json())
