"""Shared helpers for lightweight raw dataset records to ATIF JSONL.

The converter intentionally stays close to the source record. It recognizes
common chat/message shapes and tool-call encodings, but it does not route data
through ADP standardization or apply repository-wide tool normalization. Dataset
``atif_to_std.py`` scripts own that later normalization stage.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from schema.atif import (
    ATIF_SCHEMA_VERSION,
    Agent,
    ATIFObservation,
    ATIFTrajectory,
    ContentPart,
    ImageSource,
    ObservationResult,
    Step,
    ToolCall,
)

ROLE_MAP = {
    "assistant": "agent",
    "agent": "agent",
    "ai": "agent",
    "bard": "agent",
    "chatgpt": "agent",
    "gpt": "agent",
    "model": "agent",
    "system": "system",
    "developer": "system",
    "human": "user",
    "instructor": "user",
    "user": "user",
    "tool_use": "agent",
}
OBSERVATION_ROLES = {"environment", "observation", "tool", "function", "tool_result"}
MESSAGE_FIELDS = ("messages", "conversations", "trajectory", "turns")
STRINGIFIED_MESSAGE_FIELDS = ("messages", "trajectory", "stitched", "full")
ID_FIELDS = (
    "id",
    "trajectory_id",
    "instance_id",
    "session_id",
    "sample_name",
    "task_stamp",
    "_id",
    "qid",
    "episode_id",
    "annotation_id",
    "unique_id",
    "task_id",
    "shortcode",
)
PROMPT_FIELDS = (
    "question",
    "prompt",
    "task",
    "goal",
    "instruction",
    "confirmed_task",
    "intent",
    "problem_statement",
    "description",
    "sop",
)
RESPONSE_FIELDS = ("response", "answer", "output", "solution")
TERMINAL_TASK_DESCRIPTION_MARKER = "\n\nTask Description:\n"


def dataset_name_from_script(script_file: str) -> str:
    return Path(script_file).resolve().parent.name


def maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def compact_extra(record: dict[str, Any]) -> dict[str, Any]:
    skipped = set(MESSAGE_FIELDS) | set(STRINGIFIED_MESSAGE_FIELDS) | {"conversations"}
    extra = {key: value for key, value in record.items() if key not in skipped}
    return json.loads(json.dumps(extra, ensure_ascii=False, default=str))


def record_id(record: Any, index: int, dataset_name: str) -> str:
    if isinstance(record, dict):
        if dataset_name == "codescout":
            instance_id = record.get("instance_id")
            source_config = record.get("source_config")
            if instance_id and source_config:
                return (
                    f"codescout_{source_config}_{instance_id}_step{record.get('step', 0)}"
                    f"_rollout{record.get('rollout_number', 0)}"
                )
            if record.get("source_config") and record.get("source_split") is not None:
                return (
                    f"codescout_{record['source_config']}_{record['source_split']}_"
                    f"{record.get('row_id', index)}"
                )
        if (
            dataset_name == "nemotron_terminal_corpus"
            and record.get("task")
            and record.get("episode")
        ):
            return f"{record['task']}_{record['episode']}"
        if dataset_name == "litecoder-terminal-sft" and record.get("id") is not None:
            return f"litecoder-terminal-sft-{record['id']}"
        if dataset_name == "openresearcher" and record.get("qid") is not None:
            config = record.get("config") or "seed_42"
            split = record.get("split") or "train"
            return f"{config}_{split}_{record['qid']}"
        if dataset_name == "go-browse-wa" and isinstance(record.get("traj_data"), dict):
            return str(record["traj_data"].get("traj_num", index))
        for field in ID_FIELDS:
            if field in record and record[field] is not None:
                return str(record[field])
    return str(index)


def image_path(part: dict[str, Any]) -> str:
    source = part.get("source")
    if isinstance(source, dict):
        return str(source.get("path") or source.get("url") or "image")
    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        return str(image_url.get("url") or "image")
    return str(part.get("path") or part.get("url") or "image")


def text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, ContentPart):
                parts.append(
                    part.text or f"[Image: {part.source.path if part.source else 'image'}]"
                )
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                elif "text" in part:
                    parts.append(str(part["text"]))
                elif part.get("type") in {"image", "image_url"}:
                    parts.append(f"[Image: {image_path(part)}]")
            else:
                parts.append(str(part))
        return "\n".join(part for part in parts if part)
    return str(content)


def atif_content(content: Any) -> str | list[ContentPart]:
    if isinstance(content, list):
        parts: list[ContentPart] = []
        for item in content:
            if isinstance(item, ContentPart):
                parts.append(item)
            elif isinstance(item, str):
                parts.append(ContentPart(type="text", text=item))
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(ContentPart(type="text", text=str(item.get("text", ""))))
            elif isinstance(item, dict) and item.get("type") in {"image", "image_url"}:
                parts.append(
                    ContentPart(
                        type="image",
                        source=ImageSource(
                            path=image_path(item), media_type=item.get("media_type")
                        ),
                        extra={key: value for key, value in item.items() if key not in {"type"}},
                    )
                )
        if parts:
            return parts
    return text_from_content(content)


def json_safe_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def json_safe_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return {str(key): json_safe_value(value) for key, value in arguments.items()}


def parse_arguments(value: Any) -> dict[str, Any]:
    value = maybe_json(value)
    if isinstance(value, dict):
        return json_safe_arguments(value)
    if value in (None, ""):
        return {}
    return {"value": json_safe_value(value)}


def openai_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    tool_calls = []
    for idx, call in enumerate(message.get("tool_calls") or []):
        function = call.get("function", {}) if isinstance(call, dict) else {}
        name = function.get("name") or call.get("name") or call.get("function_name")
        if not name:
            continue
        tool_calls.append(
            ToolCall(
                tool_call_id=str(call.get("id") or call.get("tool_call_id") or f"call_{idx + 1}"),
                function_name=str(name),
                arguments=parse_arguments(function.get("arguments") or call.get("arguments")),
                extra={"raw_type": call.get("type")} if call.get("type") else None,
            )
        )
    return tool_calls


def source_tool_call(message: dict[str, Any]) -> list[ToolCall]:
    name = message.get("tool_name") or message.get("function_name")
    if not name:
        return []
    raw_args = (
        message.get("tool_input")
        or message.get("tool_input_json")
        or message.get("arguments")
        or message.get("content")
    )
    args = parse_arguments(raw_args)
    original_name = str(name)
    lower_name = original_name.lower()
    if lower_name in {"bash", "shell", "execute_bash"}:
        function_name = "bash"
        arguments = {
            "command": args.get("command") or message.get("command") or args.get("value") or ""
        }
    elif lower_name in {"read", "read_file"}:
        function_name = "str_replace_editor"
        arguments = {
            "command": "view",
            "path": args.get("path") or args.get("file_path") or message.get("file_path") or "",
        }
    elif lower_name == "write":
        function_name = "str_replace_editor"
        arguments = {
            "command": "create",
            "path": args.get("path") or args.get("file_path") or message.get("file_path") or "",
            "file_text": args.get("content") or args.get("file_text") or "",
        }
    elif lower_name == "edit_file":
        function_name = "str_replace_editor"
        arguments = {
            "command": "str_replace",
            "path": args.get("path") or args.get("file_path") or message.get("file_path") or "",
            "old_str": args.get("old_str") or args.get("old_string") or "",
            "new_str": args.get("new_str") or args.get("new_string") or "",
        }
    else:
        function_name = "generic_tool"
        arguments = {"tool_name": original_name, "tool_input": args}
    return [
        ToolCall(
            tool_call_id=str(message.get("tool_call_id") or message.get("id") or "call_1"),
            function_name=function_name,
            arguments=arguments,
            extra={"raw_tool_name": original_name},
        )
    ]


def function_calls_field_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    raw_calls = message.get("function_calls")
    if not raw_calls:
        return []
    calls = raw_calls if isinstance(raw_calls, list) else str(raw_calls).splitlines()
    tool_calls = []
    for index, raw_call in enumerate(calls, start=1):
        if isinstance(raw_call, dict):
            name = raw_call.get("name") or raw_call.get("function_name")
            arguments = parse_arguments(raw_call.get("arguments"))
        else:
            match = re.match(r"\s*([A-Za-z0-9_.-]+)\((.*)\)\s*$", str(raw_call), flags=re.DOTALL)
            if not match:
                continue
            name = match.group(1)
            try:
                parsed = ast.parse(f"f({match.group(2)})", mode="eval")
                if not isinstance(parsed.body, ast.Call):
                    raise ValueError("function call arguments did not parse as a call")
                arguments = {
                    keyword.arg: ast.literal_eval(keyword.value)
                    for keyword in parsed.body.keywords
                    if keyword.arg is not None
                }
            except (AttributeError, SyntaxError, ValueError):
                arguments = {"arguments": match.group(2)}
            arguments = json_safe_arguments(arguments)
        if not name:
            continue
        tool_calls.append(
            ToolCall(
                tool_call_id=f"call_{index}",
                function_name=re.sub(r"[^A-Za-z0-9_]", "_", str(name)),
                arguments=arguments,
                extra={"raw_function_name": str(name)},
            )
        )
    return tool_calls


def xml_tool_calls(content: str) -> tuple[str, list[ToolCall]]:
    tool_calls: list[ToolCall] = []
    message = content

    execute_match = re.search(r"<execute>(.*?)</execute>", content, flags=re.DOTALL)
    if execute_match:
        thought = content[: execute_match.start()].strip()
        code = execute_match.group(1).strip()
        tool_calls.append(
            ToolCall(
                tool_call_id="call_1",
                function_name="execute",
                arguments={"code": code},
                extra={"raw_format": "execute_xml"},
            )
        )
        return thought, tool_calls

    function_match = re.search(
        r"<function=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</function>",
        content,
        flags=re.DOTALL,
    )
    if function_match:
        function_name = function_match.group(1)
        if function_name == "example_function_name":
            return message, tool_calls
        params = {
            param.group(1): param.group(2).strip()
            for param in re.finditer(
                r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</parameter>",
                function_match.group(2),
                flags=re.DOTALL,
            )
        }
        has_unnamed_parameters = re.search(
            r"<parameter>(.*?)</parameter>",
            function_match.group(2),
            flags=re.DOTALL,
        )
        if not params and (function_name == "str_replace_editor" or has_unnamed_parameters):
            return message, tool_calls
        message = (content[: function_match.start()] + content[function_match.end() :]).strip()
        tool_calls.append(
            ToolCall(
                tool_call_id="call_1",
                function_name=function_name,
                arguments=params,
                extra={"raw_format": "function_xml"},
            )
        )
    return message, tool_calls


def fenced_command_tool_calls(content: str) -> tuple[str, list[ToolCall]] | None:
    match = re.search(r"```(?:bash|sh)?\s*(.*?)\s*```", content, flags=re.DOTALL)
    if not match:
        return None
    command = match.group(1).strip()
    if not command:
        return None
    thought = (content[: match.start()] + content[match.end() :]).strip()
    return thought, [
        ToolCall(
            tool_call_id="call_1",
            function_name="bash",
            arguments={"command": command},
        )
    ]


def json_command_tool_calls(content: str) -> tuple[str, list[ToolCall]] | None:
    json_match = re.search(r"(\{.*\})\s*$", content, flags=re.DOTALL)
    if not json_match:
        return None
    try:
        payload = json.loads(json_match.group(1))
    except json.JSONDecodeError:
        return None
    commands = payload.get("commands")
    if not isinstance(commands, list):
        return None
    tool_calls = []
    for index, command in enumerate(commands, start=1):
        if not isinstance(command, dict):
            continue
        keystrokes = str(command.get("keystrokes") or "").strip()
        if not keystrokes:
            continue
        tool_calls.append(
            ToolCall(
                tool_call_id=f"call_{index}",
                function_name="bash",
                arguments={"command": keystrokes},
            )
        )
    if not tool_calls:
        return None
    message_parts = [
        str(payload.get("analysis") or "").strip(),
        str(payload.get("plan") or "").strip(),
    ]
    message = "\n\n".join(part for part in message_parts if part)
    return message, tool_calls


def alfworld_action_call(action: str) -> ToolCall | None:
    action = action.strip().rstrip(".")
    lower_action = action.lower()
    function_name = None
    arguments: dict[str, Any] = {}

    if lower_action.startswith("go to "):
        function_name = "go"
        arguments = {"location": action[6:].strip()}
    elif lower_action.startswith("take ") and " from " in lower_action:
        match = re.match(r"take\s+(.*?)\s+from\s+(.*)", action, flags=re.IGNORECASE)
        if match:
            function_name = "take"
            arguments = {"item": match.group(1).strip(), "source": match.group(2).strip()}
    elif lower_action.startswith("put ") and re.search(r"\s+in/on\s+", lower_action):
        match = re.match(r"put\s+(.*?)\s+in/on\s+(.*)", action, flags=re.IGNORECASE)
        if match:
            function_name = "put"
            arguments = {"item": match.group(1).strip(), "target": match.group(2).strip()}
    elif lower_action in {"inventory", "look"}:
        function_name = lower_action
    elif lower_action.startswith("open "):
        function_name = "open"
        arguments = {"obj": action[5:].strip()}
    elif lower_action.startswith("close "):
        function_name = "close"
        arguments = {"obj": action[6:].strip()}
    elif lower_action.startswith("examine "):
        function_name = "examine"
        arguments = {"obj": action[8:].strip()}
    elif lower_action.startswith("use "):
        function_name = "use"
        arguments = {"obj": action[4:].strip()}

    if function_name is None:
        return None
    return ToolCall(tool_call_id="call_1", function_name=function_name, arguments=arguments)


def bracket_action_call(action: str) -> ToolCall | None:
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\[(.*)\]", action.strip(), flags=re.DOTALL)
    if not match:
        return None
    function_name = match.group(1)
    value = match.group(2).strip()
    if function_name == "search":
        arguments = {"keywords": value}
    elif function_name == "click":
        arguments = {"element": value}
    else:
        arguments = {"value": value}
    return ToolCall(tool_call_id="call_1", function_name=function_name, arguments=arguments)


def os_action_call(text: str) -> tuple[str, list[ToolCall]] | None:
    think_match = re.search(r"Think:\s*(.*?)(?:\n\nAct:|\nAct:)", text, flags=re.DOTALL)
    thought = think_match.group(1).strip() if think_match else ""
    bash_match = re.search(r"Act:\s*bash\s*```bash\s*(.*?)\s*```", text, flags=re.DOTALL)
    if bash_match:
        return thought, [
            ToolCall(
                tool_call_id="call_1",
                function_name="bash",
                arguments={"command": bash_match.group(1).strip()},
            )
        ]
    answer_match = re.search(r"Act:\s*answer\((.*?)\)\s*$", text, flags=re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        message = "\n\n".join(part for part in [thought, f"<finish> {answer} </finish>"] if part)
        return message, []
    finish_match = re.search(r"Act:\s*finish\s*$", text, flags=re.DOTALL)
    if finish_match:
        message = "\n\n".join(part for part in [thought, "<finish> done </finish>"] if part)
        return message, []
    return None


def source_format_tool_calls(content: str) -> tuple[str, list[ToolCall]]:
    os_call = os_action_call(content)
    if os_call is not None:
        return os_call

    fenced_call = fenced_command_tool_calls(content)
    if fenced_call is not None:
        return fenced_call

    json_command_call = json_command_tool_calls(content)
    if json_command_call is not None:
        return json_command_call

    action_match = re.search(r"(?:^|\n)\s*ACTION:\s*(.*?)\s*$", content, flags=re.DOTALL)
    if action_match:
        action = action_match.group(1).strip()
        thought = content[: action_match.start()].strip()
        thought = re.sub(r"^THOUGHT:\s*", "", thought, flags=re.IGNORECASE).strip()
        tool_call = alfworld_action_call(action) or bracket_action_call(action)
        if tool_call is not None:
            return thought, [tool_call]

    webshop_match = re.search(r"(?:^|\n)\s*Action:\s*(.*?)\s*$", content, flags=re.DOTALL)
    if webshop_match:
        action = webshop_match.group(1).strip()
        thought = content[: webshop_match.start()].strip()
        thought = re.sub(r"^Thought:\s*", "", thought, flags=re.IGNORECASE).strip()
        tool_call = bracket_action_call(action)
        if tool_call is not None:
            return thought, [tool_call]

    return content, []


def message_role(message: dict[str, Any]) -> str | None:
    raw_role = (
        message.get("role")
        or message.get("from")
        or message.get("speaker")
        or message.get("source")
    )
    if raw_role is None:
        return None
    raw_role = str(raw_role).lower()
    if raw_role in OBSERVATION_ROLES:
        return "observation"
    return ROLE_MAP.get(raw_role)


def message_content(message: dict[str, Any]) -> Any:
    if "content" in message:
        return message["content"]
    if "value" in message:
        return message["value"]
    if "utterance" in message:
        return message["utterance"]
    if "text" in message:
        return message["text"]
    if "system_prompt" in message:
        return message["system_prompt"]
    return ""


def add_observation(steps: list[Step], content: Any, source_call_id: str | None = None) -> None:
    result_content = atif_content(content)
    if isinstance(result_content, str) and result_content.startswith("Observation:\n"):
        result_content = result_content.split("Observation:\n", 1)[1]
    if steps and steps[-1].source == "agent":
        if (
            source_call_id is None
            and isinstance(result_content, str)
            and steps[-1].tool_calls
            and len(steps[-1].tool_calls) > 1
        ):
            parts = [part for part in result_content.splitlines() if part.strip()]
            if len(parts) == len(steps[-1].tool_calls):
                observation = steps[-1].observation or ATIFObservation(results=[])
                for tool_call, part in zip(steps[-1].tool_calls, parts):
                    observation.results.append(
                        ObservationResult(
                            source_call_id=tool_call.tool_call_id,
                            content=part,
                        )
                    )
                steps[-1].observation = observation
                return
        result = ObservationResult(source_call_id=source_call_id, content=result_content)
        if source_call_id is None and steps[-1].tool_calls:
            result.source_call_id = steps[-1].tool_calls[0].tool_call_id
        observation = steps[-1].observation or ATIFObservation(results=[])
        observation.results.append(result)
        steps[-1].observation = observation
        return
    steps.append(
        Step(
            step_id=len(steps) + 1,
            source="agent",
            message="",
            observation=ATIFObservation(results=[ObservationResult(content=result_content)]),
            llm_call_count=0,
        )
    )


def append_message_step(steps: list[Step], message: dict[str, Any]) -> None:
    role = message_role(message)
    content = message_content(message)
    if role is None:
        return
    if role == "observation" or (
        role == "user" and text_from_content(content).startswith("Observation:\n")
    ):
        add_observation(steps, content, message.get("tool_call_id"))
        return
    if role == "user" and steps and steps[-1].source == "agent" and steps[-1].tool_calls:
        add_observation(steps, content, message.get("tool_call_id"))
        return
    if role in {"system", "user"}:
        steps.append(Step(step_id=len(steps) + 1, source=role, message=atif_content(content)))
        return

    text = text_from_content(content)
    tool_calls = openai_tool_calls(message)
    if not tool_calls:
        tool_calls = function_calls_field_tool_calls(message)
    if not tool_calls:
        tool_calls = source_tool_call(message)
        if tool_calls and message.get("tool_name"):
            text = str(message["tool_name"])
    if not tool_calls and message.get("turn_type") == "assistant_thinking" and text:
        tool_calls = [
            ToolCall(
                tool_call_id=str(message.get("tool_call_id") or "call_1"),
                function_name="think",
                arguments={"thought": text},
            )
        ]
        text = ""
    if not tool_calls and text:
        text, tool_calls = xml_tool_calls(text)
    if not tool_calls and text:
        text, tool_calls = source_format_tool_calls(text)
    steps.append(
        Step(
            step_id=len(steps) + 1,
            source="agent",
            message=atif_content(text),
            tool_calls=tool_calls or None,
        )
    )


def find_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    chat_messages = record.get("chat_messages")
    if isinstance(chat_messages, dict) and isinstance(chat_messages.get("messages"), list):
        return chat_messages["messages"]
    for field in MESSAGE_FIELDS:
        value = maybe_json(record.get(field))
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value
    for field in STRINGIFIED_MESSAGE_FIELDS:
        value = maybe_json(record.get(field))
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value
    return []


def fallback_steps(record: dict[str, Any]) -> list[Step]:
    prompt = next((record[field] for field in PROMPT_FIELDS if record.get(field)), None)
    response = next((record[field] for field in RESPONSE_FIELDS if record.get(field)), None)
    if prompt is None:
        prompt = json.dumps(record, ensure_ascii=False, default=str)
    steps = [Step(step_id=1, source="user", message=text_from_content(prompt))]
    if response is not None:
        steps.append(Step(step_id=2, source="agent", message=text_from_content(response)))
    return steps


def tool_definitions(record: dict[str, Any]) -> list[dict[str, Any]] | None:
    for field in ("tools", "available_tools"):
        value = maybe_json(record.get(field))
        if isinstance(value, list):
            return value
    chat_messages = record.get("chat_messages")
    if isinstance(chat_messages, dict):
        value = maybe_json(chat_messages.get("tools"))
        if isinstance(value, list):
            return value
    return None


def trajectory_from_record(record: dict[str, Any], index: int, dataset_name: str) -> ATIFTrajectory:
    steps: list[Step] = []
    for message in find_messages(record):
        append_message_step(steps, message)
    if not steps:
        steps = fallback_steps(record)
    trajectory_id = record_id(record, index, dataset_name)
    agent_name = str(record.get("agent") or record.get("model") or dataset_name)
    return ATIFTrajectory(
        schema_version=ATIF_SCHEMA_VERSION,
        session_id=str(record.get("session_id")) if record.get("session_id") else trajectory_id,
        trajectory_id=trajectory_id,
        agent=Agent(
            name=agent_name,
            version=str(record.get("version") or record.get("model_name") or "raw"),
            model_name=str(record.get("model") or record.get("model_name"))
            if record.get("model") or record.get("model_name")
            else None,
            tool_definitions=tool_definitions(record),
        ),
        steps=renumber_steps(steps),
        final_metrics={"reward": record.get("reward")}
        if isinstance(record.get("reward"), int | float | bool)
        else None,
        extra={"raw": compact_extra(record), "source_dataset": dataset_name},
    )


def renumber_steps(steps: list[Step]) -> list[Step]:
    for index, step in enumerate(steps, start=1):
        step.step_id = index
    return steps


def split_terminal_task_description_prompt(trajectory: ATIFTrajectory) -> bool:
    if not trajectory.steps:
        return False
    first_step = trajectory.steps[0]
    if first_step.source != "user" or not isinstance(first_step.message, str):
        return False
    if TERMINAL_TASK_DESCRIPTION_MARKER not in first_step.message:
        return False
    system_prompt, task_prompt = first_step.message.split(
        TERMINAL_TASK_DESCRIPTION_MARKER, 1
    )
    first_step.source = "system"
    first_step.message = system_prompt.strip()
    trajectory.steps.insert(
        1,
        Step(
            step_id=2,
            source="user",
            message=f"Task Description:\n{task_prompt.strip()}",
        ),
    )
    trajectory.steps = renumber_steps(trajectory.steps)
    return True


def structure_terminal_completion_step(step: Step) -> bool:
    if step.source != "agent" or step.tool_calls or step.observation is not None:
        return False
    text = text_from_content(step.message).strip()
    if not text.startswith("{"):
        return False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict) or payload.get("task_complete") is not True:
        return False
    commands = payload.get("commands")
    if not isinstance(commands, list) or any(
        isinstance(command, dict) and str(command.get("keystrokes") or "").strip()
        for command in commands
    ):
        return False
    message_parts = [
        str(payload.get("analysis") or "").strip(),
        str(payload.get("plan") or "").strip(),
    ]
    step.message = "\n\n".join(part for part in message_parts if part)
    step.tool_calls = [
        ToolCall(
            tool_call_id="call_1",
            function_name="finish",
            arguments={"message": step.message, "task_completed": True},
            extra={"raw_format": "terminal_json"},
        )
    ]
    return True


def screenagent_trajectories(items: list[Any], dataset_name: str) -> Iterable[ATIFTrajectory]:
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        session_id = item.get("session_id") or "screenagent"
        steps = [
            Step(
                step_id=1,
                source="user",
                message=str(item.get("task_prompt_en") or item.get("task_prompt") or item),
            )
        ]
        screenshot = item.get("screenshot")
        if screenshot and screenshot != "<image>":
            add_observation(steps, [ContentPart(type="image", source=ImageSource(path=screenshot))])
        response = (
            item.get("LLM_response_editer_en") or item.get("LLM_response") or item.get("response")
        )
        if response:
            steps.append(Step(step_id=len(steps) + 1, source="agent", message=str(response)))
        yield ATIFTrajectory(
            trajectory_id=f"{session_id}_{index}",
            session_id=str(session_id),
            agent=Agent(name=dataset_name, version="raw"),
            steps=renumber_steps(steps),
            extra={"raw": item, "source_dataset": dataset_name},
        )


def trajectories_from_input(records: Iterable[Any], dataset_name: str) -> Iterable[ATIFTrajectory]:
    for index, record in enumerate(records):
        if isinstance(record, list):
            yield from screenagent_trajectories(record, dataset_name)
        elif isinstance(record, dict):
            yield trajectory_from_record(record, index, dataset_name)
        else:
            yield ATIFTrajectory(
                trajectory_id=str(index),
                agent=Agent(name=dataset_name, version="raw"),
                steps=[Step(step_id=1, source="user", message=str(record))],
                extra={"source_dataset": dataset_name},
            )


def main(script_file: str) -> None:
    dataset_name = dataset_name_from_script(script_file)
    records = (json.loads(line) for line in sys.stdin if line.strip())
    for trajectory in trajectories_from_input(records, dataset_name):
        print(trajectory.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
