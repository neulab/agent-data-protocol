from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Sequence
from typing import Any

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
os.environ.setdefault("LOG_LEVEL", "ERROR")

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    LLMConvertibleEvent,
    ToolDefinition,
)
from openhands.sdk import (
    Action as SDKAction,
)
from openhands.sdk import (
    Observation as SDKObservation,
)
from openhands.sdk.event import ActionEvent, MessageEvent, ObservationEvent
from openhands.sdk.llm import ImageContent, Message, MessageToolCall, TextContent
from openhands.sdk.tool import Tool, ToolExecutor, register_tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool
from openhands.tools.terminal.definition import MAX_CMD_OUTPUT_SIZE, maybe_truncate
from pydantic import SecretStr

from schema.dataset_metadata import (
    DatasetMetadata,
    OpenAIToolSpec,
    custom_tool_map,
    is_browser_api_action,
    load_dataset_metadata,
)
from scripts.atif_input import (
    ApiAction,
    CodeAction,
    ImageObservation,
    MessageAction,
    TextObservation,
    Trajectory,
    WebObservation,
    load_trajectory,
)

try:
    from openhands.tools.browser_use import BrowserToolSet
except Exception:  # noqa: BLE001
    BrowserToolSet = None

_REGISTERED_METADATA_TOOL_SPECS: dict[str, dict[str, Any]] = {}

BROWSER_TOOL_ALIASES = {
    "back": "browser_go_back",
    "click": "browser_click",
    "fill": "browser_type",
    "go_back": "browser_go_back",
    "goto": "browser_navigate",
    "scroll": "browser_scroll",
    "type": "browser_type",
}
BROWSER_INDEX_KWARG_NAMES = {"bid", "element_id", "id", "index"}

OPENHANDS_TOOL_ALIASES = {
    "bash": "terminal",
    "edit_file": "file_editor",
    "execute_bash": "terminal",
    "finish": "finish",
    "stop": "finish",
    "str_replace_editor": "file_editor",
    "submit": "finish",
    "task_tracker": "task_tracker",
    "think": "think",
}


class DatasetToolObservation(SDKObservation):
    output: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        output = maybe_truncate(
            content=self.output,
            truncate_after=MAX_CMD_OUTPUT_SIZE,
            tool_prefix="dataset_tool",
        )
        return [TextContent(text=output)]


class DatasetToolExecutor(ToolExecutor):
    def __call__(
        self,
        action: SDKAction,  # noqa: ARG002
        conversation: Conversation | None = None,  # noqa: ARG002
    ) -> DatasetToolObservation:
        return DatasetToolObservation(output="")


def parse_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def normalize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: parse_scalar(value) for key, value in kwargs.items()}


def _schema_type(schema: dict[str, Any]) -> str | None:
    schema_type = schema.get("type")
    return schema_type if isinstance(schema_type, str) else None


def coerce_value_for_schema(value: Any, schema: dict[str, Any]) -> Any:
    schema_type = _schema_type(schema)
    if schema_type == "string":
        return stringify_value(value)
    if schema_type in {"integer", "number", "boolean", "array", "object"}:
        return parse_scalar(value)
    for option in schema.get("anyOf", []) or []:
        if isinstance(option, dict) and _schema_type(option) == "string" and isinstance(value, str):
            return value
    return value


def normalize_metadata_tool_kwargs(
    function_name: str, kwargs: dict[str, Any], metadata: DatasetMetadata
) -> dict[str, Any]:
    tool_spec = custom_tool_map(metadata).get(function_name)
    if tool_spec is None:
        return normalize_kwargs(kwargs)
    properties = (tool_spec.function.parameters or {}).get("properties", {}) or {}
    return {
        key: coerce_value_for_schema(value, properties.get(key, {}))
        for key, value in kwargs.items()
    }


FILE_EDITOR_STRING_FIELDS = {"command", "path", "file_text", "old_str", "new_str"}


def stringify_value(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def normalize_file_editor_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in FILE_EDITOR_STRING_FIELDS:
            normalized[key] = stringify_value(value)
        else:
            normalized[key] = parse_scalar(value)
    return normalized


SHELL_CODE_LANGUAGES = {"bash", "sh", "shell"}
HEREDOC_INTERPRETERS = {
    "python": "python",
    "python3": "python",
    "py": "python",
}
SUPPORTED_TERMINAL_CODE_LANGUAGES = SHELL_CODE_LANGUAGES | set(HEREDOC_INTERPRETERS)


def class_name(name: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", name)
    text = "".join(part[:1].upper() + part[1:] for part in parts if part)
    if not text or text[0].isdigit():
        text = "Dataset" + text
    return text


def normalize_parameters(function: OpenAIToolSpec) -> dict[str, Any]:
    parameters = function.function.parameters or {}
    if parameters.get("type") != "object":
        parameters = {**parameters, "type": "object"}
    parameters.setdefault("properties", {})
    return parameters


def make_metadata_tool(name: str, tool_spec: OpenAIToolSpec) -> type[ToolDefinition]:
    parameters = normalize_parameters(tool_spec)
    action_type = SDKAction.from_mcp_schema(f"{class_name(name)}Action", parameters)
    description = tool_spec.function.description or f"Dataset metadata tool {name}."

    def create(
        cls,
        conv_state=None,  # noqa: ARG001
        _description=description,
        _action_type=action_type,
        **params,  # noqa: ARG001
    ) -> list[ToolDefinition]:
        return [
            cls(
                description=_description,
                action_type=_action_type,
                observation_type=DatasetToolObservation,
                executor=DatasetToolExecutor(),
            )
        ]

    return type(
        f"{class_name(name)}Tool",
        (ToolDefinition,),
        {"name": name, "create": classmethod(create)},
    )


def serializable_metadata_tool_spec(tool_spec: OpenAIToolSpec) -> dict[str, Any]:
    return tool_spec.model_dump(mode="json")


def register_metadata_tools(metadata: DatasetMetadata) -> None:
    for name, tool_spec in custom_tool_map(metadata).items():
        serialized_spec = serializable_metadata_tool_spec(tool_spec)
        registered_spec = _REGISTERED_METADATA_TOOL_SPECS.get(name)
        if registered_spec is not None:
            if registered_spec != serialized_spec:
                raise ValueError(
                    f"Metadata custom tool {name!r} was already registered with a "
                    "different schema. Run one dataset per converter process or use "
                    "unique custom tool names."
                )
            continue
        register_tool(name, make_metadata_tool(name, tool_spec))
        _REGISTERED_METADATA_TOOL_SPECS[name] = serialized_spec


def available_custom_tools(trajectory: Trajectory, metadata: DatasetMetadata) -> list[str]:
    available = getattr(trajectory, "available_custom_tools", None)
    if available is None:
        available = getattr(trajectory, "available_apis", None)
    if available is None:
        return sorted(custom_tool_map(metadata))
    return list(available)


def custom_tool_uses_browser_index(tool_spec: OpenAIToolSpec) -> bool:
    properties = (tool_spec.function.parameters or {}).get("properties", {})
    return bool(BROWSER_INDEX_KWARG_NAMES & set(properties))


def sdk_tool_specs(trajectory: Trajectory, metadata: DatasetMetadata) -> list[Tool]:
    specs: list[Tool] = []
    code_languages = {language.lower() for language in metadata.code_enabled}
    if code_languages & SUPPORTED_TERMINAL_CODE_LANGUAGES:
        specs.append(Tool(name=TerminalTool.name))
    if metadata.file_editor_enabled:
        specs.append(Tool(name=FileEditorTool.name))
    unsupported_code = sorted(code_languages - SUPPORTED_TERMINAL_CODE_LANGUAGES)
    if unsupported_code:
        raise ValueError(
            "OpenHands SDK conversion only supports shell-like or directly executable "
            f"CodeAction entries. Unsupported code languages: {unsupported_code}"
        )
    if metadata.browser_enabled:
        if BrowserToolSet is None:
            raise ValueError("metadata.browser_enabled is true, but BrowserToolSet is unavailable")
        specs.append(Tool(name=BrowserToolSet.name))

    registered_custom_tools = custom_tool_map(metadata)
    for source_name in available_custom_tools(trajectory, metadata):
        mapped_name = OPENHANDS_TOOL_ALIASES.get(source_name, source_name)
        if (
            metadata.browser_enabled
            and source_name in BROWSER_TOOL_ALIASES
            and (
                source_name not in registered_custom_tools
                or custom_tool_uses_browser_index(registered_custom_tools[source_name])
            )
        ):
            continue
        if mapped_name == "terminal":
            if "bash" not in metadata.code_enabled:
                raise ValueError(f"{source_name!r} maps to terminal, but bash is disabled")
            continue
        if mapped_name == "file_editor":
            specs.append(Tool(name=FileEditorTool.name))
            continue
        if mapped_name == "task_tracker":
            specs.append(Tool(name=TaskTrackerTool.name))
            continue
        if mapped_name in {"finish", "think"}:
            continue
        if source_name not in registered_custom_tools:
            raise ValueError(f"available tool {source_name!r} is not declared in metadata.json")
        specs.append(Tool(name=source_name))
    return dedupe_tools(specs)


def dedupe_tools(tools: list[Tool]) -> list[Tool]:
    seen: set[str] = set()
    result: list[Tool] = []
    for tool in tools:
        if tool.name in seen:
            continue
        seen.add(tool.name)
        result.append(tool)
    return result


def text_message(role: str, content: str) -> Message:
    return Message(role=role, content=[TextContent(text=content)])


def event_description(event: ApiAction | CodeAction | MessageAction) -> str:
    return (getattr(event, "description", None) or "").strip()


def event_reasoning_content(event: ApiAction | CodeAction | MessageAction) -> str | None:
    reasoning = getattr(event, "reasoning_content", None)
    if reasoning:
        return str(reasoning).strip()
    return None


def extract_finish_message(content: str) -> str | None:
    match = re.search(r"<finish>(.*?)</finish>", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_legacy_tool_call(content: str) -> tuple[str, dict[str, Any], str] | None:
    match = re.search(r"<function=([A-Za-z_][A-Za-z0-9_]*)>", content)
    if not match:
        return None
    if match.group(1) == "example_function_name":
        return None
    close_match = re.search(r"</function>", content[match.end() :])
    block_end = match.end() + close_match.start() if close_match is not None else len(content)
    trailing_start = match.end() + close_match.end() if close_match is not None else len(content)
    block = content[match.end() : block_end]
    args = {
        param_match.group(1): param_match.group(2).strip()
        for param_match in re.finditer(
            r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</parameter>",
            block,
            re.DOTALL,
        )
    }
    if not args and (
        match.group(1) == "str_replace_editor"
        or re.search(r"<parameter>(.*?)</parameter>", block, re.DOTALL)
    ):
        return None
    thought = (content[: match.start()] + content[trailing_start:]).strip()
    return match.group(1), args, thought


def coerce_browser_index(value: Any) -> int:
    value = parse_scalar(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return int(match.group(0))
    return 0


def browser_index_argument(kwargs: dict[str, Any]) -> Any | None:
    for key in ("index", "bid", "id", "element_id"):
        if key in kwargs:
            return kwargs[key]
    return None


def map_browser_action(function_name: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if function_name == "goto":
        return "browser_navigate", {"url": kwargs.get("url", ""), "new_tab": False}
    if function_name in {"go_back", "back"}:
        return "browser_go_back", {}
    if function_name == "click":
        index = browser_index_argument(kwargs)
        return "browser_click", {"index": coerce_browser_index(index), "new_tab": False}
    if function_name in {"type", "fill"}:
        index = browser_index_argument(kwargs)
        text = kwargs.get("text", kwargs.get("value", ""))
        return "browser_type", {"index": coerce_browser_index(index), "text": text}
    if function_name == "scroll":
        direction = kwargs.get("direction")
        if not direction:
            delta = kwargs.get("delta_y", kwargs.get("dy", 0))
            direction = "up" if isinstance(delta, (int, float)) and delta < 0 else "down"
        return "browser_scroll", {"direction": direction if direction in {"up", "down"} else "down"}
    return BROWSER_TOOL_ALIASES[function_name], kwargs


def should_map_to_browser_action(
    function_name: str, kwargs: dict[str, Any], metadata: DatasetMetadata
) -> bool:
    if function_name not in BROWSER_TOOL_ALIASES:
        return False
    if function_name in custom_tool_map(metadata) and browser_index_argument(kwargs) is None:
        return False
    return is_browser_api_action(function_name, kwargs, browser_context=metadata.browser_enabled)


def map_api_action(event: ApiAction, metadata: DatasetMetadata) -> tuple[str, dict[str, Any]]:
    function_name = event.function
    kwargs = (
        normalize_file_editor_kwargs(event.kwargs)
        if function_name == "str_replace_editor"
        else normalize_kwargs(event.kwargs)
    )
    if should_map_to_browser_action(function_name, kwargs, metadata):
        return map_browser_action(function_name, kwargs)
    tool_name = OPENHANDS_TOOL_ALIASES.get(function_name, function_name)
    if function_name == "submit":
        return "finish", {"message": stringify_value(kwargs.get("message"), "Done.")}
    if function_name == "stop":
        return "finish", {
            "message": stringify_value(kwargs.get("output", kwargs.get("message", "")))
        }
    if tool_name == "finish":
        return "finish", {
            "message": stringify_value(kwargs.get("message", kwargs.get("output", "")))
        }
    if function_name == "edit_file":
        return "file_editor", {
            "command": "str_replace",
            "path": stringify_value(kwargs.get("path")),
            "old_str": stringify_value(kwargs.get("old_str")),
            "new_str": stringify_value(kwargs.get("content", kwargs.get("new_str"))),
        }
    if tool_name == function_name:
        return tool_name, normalize_metadata_tool_kwargs(function_name, event.kwargs, metadata)
    return tool_name, kwargs


def heredoc_command(interpreter: str, language: str, content: str) -> str:
    digest = hashlib.sha1(content.encode()).hexdigest()[:12]
    delimiter = f"ATIF_{re.sub(r'[^A-Za-z0-9]+', '_', language).upper()}_{digest}"
    return f"{interpreter} <<'{delimiter}'\n{content}\n{delimiter}"


def map_code_action(event: CodeAction) -> tuple[str, dict[str, Any]]:
    language = event.language.lower()
    if language in SHELL_CODE_LANGUAGES:
        return "terminal", {"command": event.content}
    if language in HEREDOC_INTERPRETERS:
        return "terminal", {
            "command": heredoc_command(
                HEREDOC_INTERPRETERS[language],
                language,
                event.content,
            )
        }
    raise ValueError(
        "OpenHands SDK conversion only supports shell-like or directly executable "
        f"CodeAction entries. Encountered language {event.language!r}."
    )


def tool_call_id(index: int, tool_name: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9_]+", "_", tool_name).strip("_")[:32] or "tool"
    return f"call_{index:06d}_{suffix}"


def make_action_event(
    *,
    sdk_tool: ToolDefinition,
    tool_name: str,
    args: dict[str, Any],
    thought: str,
    reasoning_content: str | None,
    explicit_id: str | None,
    call_index: int,
    llm_response_id: str,
) -> ActionEvent:
    tool_call = MessageToolCall(
        id=explicit_id or tool_call_id(call_index, tool_name),
        name=tool_name,
        arguments=json.dumps(args, ensure_ascii=False),
        origin="completion",
    )
    return ActionEvent(
        thought=[TextContent(text=thought)] if thought else [],
        reasoning_content=reasoning_content,
        action=sdk_tool.action_from_arguments(args),
        tool_name=tool_name,
        tool_call_id=tool_call.id,
        tool_call=tool_call,
        llm_response_id=llm_response_id,
    )


class SDKEventBuilder:
    def __init__(self, conversation: Conversation, metadata: DatasetMetadata):
        self.conversation = conversation
        self.metadata = metadata
        self.action_ids_by_tool_call_id: dict[str, str] = {}
        self.tool_names_by_tool_call_id: dict[str, str] = {}
        self.call_index = 0
        self.last_action_event: ActionEvent | None = None
        self.pending_tool_call_ids: list[str] = []

    @property
    def tools_map(self) -> dict[str, ToolDefinition]:
        return self.conversation.agent.tools_map

    def append(self, event: LLMConvertibleEvent) -> None:
        self.conversation.state.events.append(event)

    def append_action_batch(
        self, events: list[ApiAction | CodeAction], *, batch_number: int
    ) -> None:
        self.flush_missing_tool_results()
        response_id = f"llm_response_{batch_number:06d}"
        for offset, event in enumerate(events):
            self.call_index += 1
            if isinstance(event, ApiAction):
                tool_name, args = map_api_action(event, self.metadata)
            else:
                tool_name, args = map_code_action(event)
            if tool_name not in self.tools_map:
                raise ValueError(f"Tool {tool_name!r} was not initialized by the SDK agent")
            action_event = make_action_event(
                sdk_tool=self.tools_map[tool_name],
                tool_name=tool_name,
                args=args,
                thought=event_description(event) if offset == 0 else "",
                reasoning_content=event_reasoning_content(event) if offset == 0 else None,
                explicit_id=event.tool_call_id,
                call_index=self.call_index,
                llm_response_id=response_id,
            )
            self.action_ids_by_tool_call_id[action_event.tool_call_id] = action_event.id
            self.tool_names_by_tool_call_id[action_event.tool_call_id] = action_event.tool_name
            self.pending_tool_call_ids.append(action_event.tool_call_id)
            self.last_action_event = action_event
            self.append(action_event)

    def _append_tool_result(self, tool_call_id: str, content: str) -> None:
        action_id = self.action_ids_by_tool_call_id.get(tool_call_id)
        if action_id is None or tool_call_id not in self.pending_tool_call_ids:
            self.flush_missing_tool_results()
            self.append(
                MessageEvent(
                    source="environment",
                    llm_message=text_message("user", content),
                )
            )
            return
        tool_name = self.tool_names_by_tool_call_id[tool_call_id]
        self.append(
            ObservationEvent(
                observation=DatasetToolObservation(output=content),
                action_id=action_id,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            )
        )
        self.pending_tool_call_ids.remove(tool_call_id)

    def flush_missing_tool_results(self) -> None:
        for tool_call_id in list(self.pending_tool_call_ids):
            self._append_tool_result(tool_call_id, "")

    def append_observation(
        self, event: TextObservation | WebObservation | ImageObservation
    ) -> None:
        content = observation_content(event)
        if event.tool_call_id is None:
            self.flush_missing_tool_results()
            source = getattr(event, "source", "environment")
            role = "assistant" if source == "agent" else "user"
            self.append(MessageEvent(source=source, llm_message=text_message(role, content)))
            return
        self._append_tool_result(event.tool_call_id, content)

    def append_synthetic_tool_result(self, content: str) -> None:
        if self.last_action_event is None:
            raise ValueError("No action event is available for a synthetic tool result")
        self._append_tool_result(self.last_action_event.tool_call_id, content)


def observation_content(event: TextObservation | WebObservation | ImageObservation) -> str:
    if isinstance(event, TextObservation):
        return event.content
    if isinstance(event, WebObservation):
        parts = []
        if event.url:
            parts.append(f"URL: {event.url}")
        if event.axtree:
            parts.append(event.axtree)
        elif event.html:
            parts.append(event.html)
        return "\n\n".join(parts)
    content = f"[Image: {event.content}]"
    if event.annotations:
        annotations = [
            f"{annotation.text} ({annotation.element_type})"
            for annotation in event.annotations
            if annotation.text
        ]
        if annotations:
            content += "\nElements detected: " + ", ".join(annotations)
    return content


def append_message_action(builder: SDKEventBuilder, event: MessageAction) -> None:
    finish_message = extract_finish_message(event.content)
    if finish_message is not None:
        api_action = ApiAction(function="finish", kwargs={"message": finish_message})
        api_action.description = event.description
        builder.append_action_batch([api_action], batch_number=builder.call_index + 1)
        builder.append_synthetic_tool_result(finish_message)
        return
    if legacy_tool_call := extract_legacy_tool_call(event.content):
        function_name, kwargs, thought = legacy_tool_call
        api_action = ApiAction(function=function_name, kwargs=kwargs, description=thought)
        builder.append_action_batch([api_action], batch_number=builder.call_index + 1)
        return
    content = "\n\n".join(part for part in [event_description(event), event.content] if part)
    builder.flush_missing_tool_results()
    builder.append(
        MessageEvent(
            source="agent",
            llm_message=Message(
                role="assistant",
                content=[TextContent(text=content)],
                reasoning_content=event_reasoning_content(event),
            ),
        )
    )


def append_standardized_events(
    conversation: Conversation,
    trajectory: Trajectory,
    metadata: DatasetMetadata,
    start_index: int,
) -> None:
    builder = SDKEventBuilder(conversation, metadata)
    index = start_index
    batch_number = 0
    while index < len(trajectory.content):
        event = trajectory.content[index]
        if isinstance(event, (ApiAction, CodeAction)):
            action_batch: list[ApiAction | CodeAction] = []
            while index < len(trajectory.content) and isinstance(
                trajectory.content[index], (ApiAction, CodeAction)
            ):
                action_batch.append(trajectory.content[index])
                index += 1
            batch_number += 1
            builder.append_action_batch(action_batch, batch_number=batch_number)
            continue
        if isinstance(event, (TextObservation, WebObservation, ImageObservation)):
            builder.append_observation(event)
        elif isinstance(event, MessageAction):
            append_message_action(builder, event)
        else:
            raise ValueError(f"Unsupported event type: {type(event)}")
        index += 1
    builder.flush_missing_tool_results()


def serializable_tool(tool: ToolDefinition) -> dict[str, Any]:
    return json.loads(json.dumps(tool.to_openai_tool(), ensure_ascii=False))


def text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def normalize_message_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        normalized_message = dict(message)
        normalized_message["content"] = text_content(normalized_message.get("content", ""))
        normalized.append(normalized_message)
    return normalized


def process_trajectory(
    trajectory: Trajectory,
    model: str,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    dataset_name = dataset_name or os.getenv("MY_DATASET")
    metadata = load_dataset_metadata(dataset_name, required=True)
    register_metadata_tools(metadata)
    if not trajectory.content:
        raise ValueError("OpenHands SDK conversion expects at least one trajectory event")
    first_user_index = next(
        (
            index
            for index, event in enumerate(trajectory.content)
            if isinstance(event, TextObservation) and event.source == "user"
        ),
        None,
    )
    if first_user_index is None:
        first_user_message = ""
        start_index = 0
    else:
        first_user = trajectory.content[first_user_index]
        first_user_message = first_user.content
        start_index = first_user_index + 1
    initial_context_end = (
        first_user_index if first_user_index is not None else len(trajectory.content)
    )
    initial_context = [
        observation_content(event)
        if isinstance(event, (TextObservation, WebObservation, ImageObservation))
        else event.content
        for event in trajectory.content[:initial_context_end]
        if isinstance(event, (TextObservation, WebObservation, ImageObservation, MessageAction))
    ]
    if not first_user_message and initial_context:
        first_user_message = initial_context.pop(0)
        start_index = 1
    if initial_context:
        first_user_message = "\n\n".join(
            part
            for part in [
                first_user_message,
                "Initial context:\n" + "\n\n".join(initial_context),
            ]
            if part
        )
    if not first_user_message:
        first_user_message = f"Continue trajectory {trajectory.id}."

    llm = LLM(
        usage_id="openhands-sdk-sft-converter",
        model=model,
        api_key=SecretStr(os.getenv("LLM_API_KEY") or "not-used"),
    )
    agent = Agent(llm=llm, tools=sdk_tool_specs(trajectory, metadata))
    with tempfile.TemporaryDirectory(prefix="openhands-sdk-sft-") as tmpdir:
        conversation = Conversation(agent=agent, workspace=tmpdir, visualizer=None)
        try:
            conversation.send_message(first_user_message)
            append_standardized_events(conversation, trajectory, metadata, start_index=start_index)
            convertible_events = [
                event
                for event in conversation.state.events
                if isinstance(event, LLMConvertibleEvent)
            ]
            messages = LLMConvertibleEvent.events_to_messages(convertible_events)
            formatted_messages = normalize_message_content(llm.format_messages_for_llm(messages))
            tools = [serializable_tool(tool) for tool in agent.tools_map.values()]
        finally:
            conversation.close()
    return {
        "id": trajectory.id,
        "messages": formatted_messages,
        "tools": tools,
        "metadata": {
            "agent": "openhands_sdk",
            "format": "openai_chat_completions",
            "source_dataset": dataset_name,
            "generation": "openhands_sdk_events",
        },
    }


def process_row(line: str, model: str, dataset_name: str | None = None) -> dict[str, Any]:
    return process_trajectory(load_trajectory(line), model, dataset_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert standardized data to OpenHands SDK SFT format"
    )
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--is_web",
        choices=["yes", "no"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--api_env", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.is_web is not None or args.api_env is not None:
        raise NotImplementedError(
            "agents/openhands_sdk/std_to_sft.py reads tool configuration from "
            "datasets/$MY_DATASET/metadata.json; --is_web and --api_env are not supported."
        )
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        print(json.dumps(process_row(line, args.model), ensure_ascii=False))


if __name__ == "__main__":
    main()
