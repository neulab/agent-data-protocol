import json
import re
import sys
from typing import Any, TypeVar

from schema_raw import ConversationTurn, SchemaRaw

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.tool_call_links import create_trajectory_with_tool_call_links
from schema.trajectory import Trajectory

SHELL_TOOL_NAMES = {"bash", "run_command", "shell", "terminal", "execute_bash"}
READ_TOOL_NAMES = {"read", "read_file"}
WRITE_TOOL_NAMES = {"write", "write_file"}
EDIT_TOOL_NAMES = {"edit", "edit_file"}
THINK_TOOL_NAMES = {"think", "thinking"}
ToolEvent = TypeVar("ToolEvent", CodeAction, ApiAction, TextObservation)


def ordered_turns(data: SchemaRaw) -> list[ConversationTurn]:
    return sorted(
        data.turns,
        key=lambda item: item.turn_number if item.turn_number is not None else float("inf"),
    )


def turn_type(turn: ConversationTurn) -> str:
    return (turn.turn_type or "").lower()


def role(turn: ConversationTurn) -> str:
    return turn.role.lower()


def is_user_prompt_turn(turn: ConversationTurn) -> bool:
    return turn_type(turn) == "user_prompt" or (role(turn) == "user" and turn.is_conversational)


def is_tool_use_turn(turn: ConversationTurn) -> bool:
    return turn_type(turn) == "tool_use" or role(turn) == "tool_use"


def is_tool_result_turn(turn: ConversationTurn) -> bool:
    return turn_type(turn) == "tool_result" or role(turn) == "tool_result"


def linkable_raw_tool_call_ids(turns: list[ConversationTurn]) -> set[str]:
    action_indices: dict[str, list[int]] = {}
    result_indices: dict[str, list[int]] = {}
    first_user_index = next(
        (index for index, turn in enumerate(turns) if is_user_prompt_turn(turn)), len(turns)
    )
    for index, turn in enumerate(turns[first_user_index:], start=first_user_index):
        tool_call_id = optional_str(turn.tool_call_id)
        if not tool_call_id:
            continue
        if is_tool_use_turn(turn):
            action_indices.setdefault(tool_call_id, []).append(index)
        elif is_tool_result_turn(turn):
            result_indices.setdefault(tool_call_id, []).append(index)

    linkable_ids = set()
    for tool_call_id, actions in action_indices.items():
        results = result_indices.get(tool_call_id, [])
        if len(actions) == 1 and len(results) == 1 and actions[0] < results[0]:
            linkable_ids.add(tool_call_id)
    return linkable_ids


def attach_raw_tool_call_id(
    event: ToolEvent,
    turn: ConversationTurn,
    linkable_tool_call_ids: set[str],
) -> ToolEvent:
    tool_call_id = optional_str(turn.tool_call_id)
    if tool_call_id in linkable_tool_call_ids:
        event.tool_call_id = tool_call_id
    return event


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


def convert_tool_use(
    turn: ConversationTurn, linkable_tool_call_ids: set[str]
) -> CodeAction | ApiAction | None:
    name = (turn.tool_name or "generic_tool").strip() or "generic_tool"
    lower_name = name.lower()
    params = tool_input(turn)

    if lower_name in SHELL_TOOL_NAMES:
        command = params.get("command") or turn.command or turn.content or ""
        return attach_raw_tool_call_id(
            CodeAction(language="bash", content=str(command), description=name),
            turn,
            linkable_tool_call_ids,
        )

    if lower_name in THINK_TOOL_NAMES:
        thought = params.get("thought") or turn.content or ""
        return attach_raw_tool_call_id(
            ApiAction(function="think", kwargs={"thought": str(thought)}, description=name),
            turn,
            linkable_tool_call_ids,
        )

    editor_action = as_str_replace_editor(turn, lower_name, params)
    if editor_action:
        editor_action.description = name
        return attach_raw_tool_call_id(editor_action, turn, linkable_tool_call_ids)

    kwargs: dict[str, Any] = {"tool_name": name, "tool_input": params}
    if not params:
        kwargs["content"] = turn.content
    return attach_raw_tool_call_id(
        ApiAction(
            function="generic_tool",
            kwargs=kwargs,
            description=clean_function_name(name),
        ),
        turn,
        linkable_tool_call_ids,
    )


def convert_turn(
    turn: ConversationTurn, seen_user_prompt: bool, linkable_tool_call_ids: set[str]
) -> list[Any]:
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

    if is_tool_use_turn(turn):
        action = convert_tool_use(turn, linkable_tool_call_ids)
        return [action] if action else []

    if is_tool_result_turn(turn):
        if not content:
            return []
        observation = TextObservation(content=content, source="environment")
        return [attach_raw_tool_call_id(observation, turn, linkable_tool_call_ids)]

    if not seen_user_prompt:
        return []

    if content:
        label = turn_type or role or "metadata"
        return [TextObservation(content=f"[{label}]\n{content}", source="environment")]
    return []


def numeric_detail(value: Any) -> int | float | Any:
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def details_from_raw(data: SchemaRaw) -> dict[str, Any]:
    details = {"source": "SALT-NLP/SWE-chat", "source_config": "conversations"}
    numeric_keys = {
        "tool_call_count",
        "turn_count",
        "prompt_count",
        "agent_percentage",
        "session_success",
    }
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
            details[key] = numeric_detail(value) if key in numeric_keys else str(value)
    return details


def process_data(data: SchemaRaw) -> Trajectory | None:
    content = []
    seen_user_prompt = False
    turns = ordered_turns(data)
    linkable_tool_call_ids = linkable_raw_tool_call_ids(turns)

    for turn in turns:
        events = convert_turn(turn, seen_user_prompt, linkable_tool_call_ids)
        has_user_prompt = any(
            isinstance(event, TextObservation) and event.source == "user" for event in events
        )
        if not seen_user_prompt and not has_user_prompt:
            continue
        if has_user_prompt:
            seen_user_prompt = True
        content.extend(events)

    if not content or not seen_user_prompt:
        return None

    return create_trajectory_with_tool_call_links(
        id=data.session_id, content=content, details=details_from_raw(data)
    )


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        if standardized_data:
            print(standardized_data.model_dump_json())
