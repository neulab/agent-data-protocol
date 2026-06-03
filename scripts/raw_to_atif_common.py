"""Shared helpers for lightweight raw dataset records to ATIF JSONL.

The converter intentionally stays close to the source record. It recognizes
common chat/message shapes and tool-call encodings, but it does not route data
through ADP standardization or apply repository-wide tool normalization. Dataset
``atif_to_std.py`` scripts own that later normalization stage.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    "gpt": "agent",
    "model": "agent",
    "system": "system",
    "developer": "system",
    "human": "user",
    "instructor": "user",
    "user": "user",
}
OBSERVATION_ROLES = {"environment", "observation", "tool", "function"}
MESSAGE_FIELDS = ("messages", "conversations", "trajectory")
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


def parse_arguments(value: Any) -> dict[str, Any]:
    value = maybe_json(value)
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return {}
    return {"value": value}


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
        params = {
            param.group(1): param.group(2).strip()
            for param in re.finditer(
                r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</parameter>",
                function_match.group(2),
                flags=re.DOTALL,
            )
        }
        message = (content[: function_match.start()] + content[function_match.end() :]).strip()
        tool_calls.append(
            ToolCall(
                tool_call_id="call_1",
                function_name=function_match.group(1),
                arguments=params,
                extra={"raw_format": "function_xml"},
            )
        )
    return message, tool_calls


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
    if "system_prompt" in message:
        return message["system_prompt"]
    return ""


def add_observation(steps: list[Step], content: Any, source_call_id: str | None = None) -> None:
    result_content = atif_content(content)
    if isinstance(result_content, str) and result_content.startswith("Observation:\n"):
        result_content = result_content.split("Observation:\n", 1)[1]
    result = ObservationResult(source_call_id=source_call_id, content=result_content)
    if steps and steps[-1].source == "agent":
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
    if role in {"system", "user"}:
        steps.append(Step(step_id=len(steps) + 1, source=role, message=atif_content(content)))
        return

    text = text_from_content(content)
    tool_calls = openai_tool_calls(message)
    if not tool_calls and text:
        text, tool_calls = xml_tool_calls(text)
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


def trajectories_from_input(records: list[Any], dataset_name: str) -> Iterable[ATIFTrajectory]:
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
    records = [json.loads(line) for line in sys.stdin if line.strip()]
    for trajectory in trajectories_from_input(records, dataset_name):
        print(trajectory.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
