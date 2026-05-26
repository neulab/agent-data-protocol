import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.dataset_metadata import (
    custom_tool_map,
    is_browser_api_action,
    load_dataset_metadata,
    validate_trajectory_metadata,
)
from schema.observation.image import ImageObservation
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory

dataset = os.getenv("MY_DATASET")

SYSTEM_PROMPT_PATH = Path(__file__).with_name("system_prompt.txt")

OPENHANDS_TOOL_ALIASES = {
    "execute_bash": "terminal",
    "bash": "terminal",
    "str_replace_editor": "file_editor",
    "edit_file": "file_editor",
    "finish": "finish",
    "submit": "finish",
    "stop": "finish",
    "think": "think",
}

BROWSER_TOOL_ALIASES = {
    "goto": "browser_navigate",
    "go_back": "browser_go_back",
    "back": "browser_go_back",
    "click": "browser_click",
    "type": "browser_type",
    "fill": "browser_type",
    "scroll": "browser_scroll",
}

READ_ONLY_TOOLS = {
    "file_editor:view",
    "task_tracker:view",
    "browser_get_content",
    "browser_get_state",
    "find",
    "open",
    "search",
    "think",
}

READ_ONLY_TERMINAL_COMMANDS = {
    "awk",
    "cat",
    "env",
    "find",
    "git",
    "grep",
    "head",
    "ls",
    "pwd",
    "rg",
    "sed",
    "tail",
    "tree",
    "wc",
    "which",
}


def _json_schema(schema_type: str, description: str | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": schema_type}
    if description:
        schema["description"] = description
    return schema


def _openai_tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
    include_security_risk: bool = True,
) -> dict[str, Any]:
    properties = {
        "summary": {
            "type": "string",
            "description": "Concise summary of what this action does.",
        },
        **properties,
    }
    if include_security_risk:
        properties = {
            "security_risk": {
                "type": "string",
                "description": "Security risk for the action.",
                "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"],
            },
            **properties,
        }
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
        },
    }


BUILTIN_TOOLS: dict[str, dict[str, Any]] = {
    "terminal": _openai_tool(
        "terminal",
        "Execute a shell command in the terminal within a persistent shell session.",
        {
            "command": _json_schema("string", "The shell command or terminal input."),
            "is_input": _json_schema(
                "boolean", "Whether command should be sent as input to a running process."
            ),
            "timeout": _json_schema("number", "Optional timeout in seconds."),
            "reset": _json_schema("boolean", "Whether to reset the terminal session."),
        },
        ["command"],
    ),
    "file_editor": _openai_tool(
        "file_editor",
        "View, create, and edit files in plain-text format.",
        {
            "command": {
                "type": "string",
                "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
            },
            "path": _json_schema("string", "Absolute file path."),
            "file_text": _json_schema("string", "Content for create operations."),
            "old_str": _json_schema("string", "Text to replace."),
            "new_str": _json_schema("string", "Replacement text."),
            "insert_line": _json_schema("integer", "Line before which to insert."),
            "view_range": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Optional inclusive line range for view.",
            },
        },
        ["command", "path"],
    ),
    "finish": _openai_tool(
        "finish",
        "Signals completion of the current task or conversation.",
        {"message": _json_schema("string", "Final message to send to the user.")},
        ["message"],
        include_security_risk=False,
    ),
    "think": _openai_tool(
        "think",
        "Log a thought without obtaining new information or changing the environment.",
        {"thought": _json_schema("string", "The thought to log.")},
        ["thought"],
        include_security_risk=False,
    ),
    "task_tracker": _openai_tool(
        "task_tracker",
        "Create, update, and inspect the task list for the current conversation.",
        {
            "command": {
                "type": "string",
                "enum": ["plan", "update", "view"],
            },
            "task_list": _json_schema("string", "Markdown task list content."),
        },
    ),
    "browser_navigate": _openai_tool(
        "browser_navigate",
        "Navigate to a URL in the browser.",
        {
            "url": _json_schema("string", "URL to navigate to."),
            "new_tab": _json_schema("boolean", "Whether to open in a new tab."),
        },
        ["url"],
    ),
    "browser_click": _openai_tool(
        "browser_click",
        "Click an element on the page by its current browser state index.",
        {
            "index": _json_schema("integer", "Index from the browser state output."),
            "new_tab": _json_schema("boolean", "Whether to open navigation in a new tab."),
        },
        ["index"],
    ),
    "browser_type": _openai_tool(
        "browser_type",
        "Type text into an input element.",
        {
            "index": _json_schema("integer", "Index from the browser state output."),
            "text": _json_schema("string", "Text to type."),
        },
        ["index", "text"],
    ),
    "browser_scroll": _openai_tool(
        "browser_scroll",
        "Scroll the page up or down.",
        {"direction": {"type": "string", "enum": ["up", "down"]}},
        ["direction"],
    ),
    "browser_go_back": _openai_tool(
        "browser_go_back",
        "Go back to the previous browser page.",
        {},
        [],
    ),
    "browser_get_state": _openai_tool(
        "browser_get_state",
        "Get current browser page state and interactive elements.",
        {"include_screenshot": _json_schema("boolean", "Whether to include a screenshot.")},
        [],
    ),
    "browser_get_content": _openai_tool(
        "browser_get_content",
        "Extract the current page content in markdown.",
        {
            "extract_links": _json_schema("boolean", "Whether to include links."),
            "start_from_char": _json_schema("integer", "Character offset to continue from."),
        },
        [],
    ),
}


BROWSER_BUILTIN_TOOL_NAMES = sorted(name for name in BUILTIN_TOOLS if name.startswith("browser_"))
OPENHANDS_SDK_NATIVE_API_NAMES = set(OPENHANDS_TOOL_ALIASES)


def text_content(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def load_system_prompt(path: str | None) -> str:
    prompt_path = Path(path) if path else SYSTEM_PROMPT_PATH
    return prompt_path.read_text()


def parse_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def normalize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: parse_scalar(value) for key, value in kwargs.items()}


def extract_finish_message(content: str) -> str | None:
    match = re.search(r"<finish>(.*?)</finish>", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_legacy_tool_call(content: str) -> tuple[str, dict[str, Any], str] | None:
    match = re.search(r"<function=([A-Za-z_][A-Za-z0-9_]*)>", content)
    if not match:
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
    thought = (content[: match.start()] + content[trailing_start:]).strip()
    return match.group(1), args, thought


def action_content(event: ApiAction | CodeAction | MessageAction) -> str:
    parts = []
    description = getattr(event, "description", None)
    if description:
        parts.append(str(description).strip())
    return "\n\n".join(part for part in parts if part)


def event_reasoning_content(event: ApiAction | CodeAction | MessageAction) -> str | None:
    reasoning = getattr(event, "reasoning_content", None)
    if reasoning:
        return str(reasoning).strip()
    return None


def safe_tool_suffix(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")[:32] or "tool"


def tool_call_id(index: int, name: str) -> str:
    return f"call_{index:06d}_{safe_tool_suffix(name)}"


def coerce_browser_index(value: Any) -> int:
    value = parse_scalar(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return int(match.group(0))
    return 0


def map_browser_action(function_name: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if function_name == "goto":
        return "browser_navigate", {"url": kwargs.get("url", ""), "new_tab": False}
    if function_name in {"go_back", "back"}:
        return "browser_go_back", {}
    if function_name == "click":
        index = kwargs.get("index", kwargs.get("bid", kwargs.get("id", kwargs.get("element_id"))))
        if index is None:
            index = kwargs.get("xpath", "")
        return "browser_click", {"index": coerce_browser_index(index)}
    if function_name in {"type", "fill"}:
        index = kwargs.get("index", kwargs.get("bid", kwargs.get("id", kwargs.get("element_id"))))
        if index is None:
            index = kwargs.get("xpath", "")
        text = kwargs.get("text", kwargs.get("value", ""))
        return "browser_type", {"index": coerce_browser_index(index), "text": text}
    if function_name == "scroll":
        direction = kwargs.get("direction")
        if not direction:
            delta = kwargs.get("delta_y", kwargs.get("dy", 0))
            direction = "up" if isinstance(delta, (int, float)) and delta < 0 else "down"
        return "browser_scroll", {"direction": direction if direction in {"up", "down"} else "down"}
    return BROWSER_TOOL_ALIASES[function_name], kwargs


def map_api_action(event: ApiAction, *, browser_action: bool = False) -> tuple[str, dict[str, Any]]:
    function_name = event.function
    kwargs = normalize_kwargs(event.kwargs)
    if browser_action and function_name in BROWSER_TOOL_ALIASES:
        return map_browser_action(function_name, kwargs)
    tool_name = OPENHANDS_TOOL_ALIASES.get(function_name, function_name)
    if function_name == "submit":
        return "finish", {"message": kwargs.get("message", "Done.")}
    if function_name == "stop":
        return "finish", {"message": kwargs.get("output", kwargs.get("message", ""))}
    if tool_name == "finish":
        return "finish", {"message": kwargs.get("message", kwargs.get("output", ""))}
    if function_name == "edit_file":
        return "file_editor", {
            "command": "str_replace",
            "path": kwargs.get("path", ""),
            "old_str": kwargs.get("old_str", ""),
            "new_str": kwargs.get("content", kwargs.get("new_str", "")),
        }
    return tool_name, kwargs


def map_code_action(event: CodeAction) -> tuple[str, dict[str, Any]]:
    language = event.language
    if language == "bash":
        return "terminal", {"command": event.content}
    raise ValueError(
        "OpenHands SDK SFT conversion only supports bash CodeAction entries. "
        f"Encountered language {language!r}."
    )


def supports_security_risk(tool_name: str) -> bool:
    return tool_name not in {"finish", "think"}


def terminal_command_is_read_only(command: Any) -> bool:
    if not isinstance(command, str) or not command.strip():
        return False
    if re.search(r"(^|[;&|]\s*)(rm|mv|cp|touch|mkdir|rmdir|chmod|chown|pip|uv)\b", command):
        return False
    if re.search(r">|>>|<<|tee\s", command):
        return False

    pieces = re.split(r"\s*(?:&&|\|\||;|\n)\s*", command)
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        first = piece.split(maxsplit=1)[0]
        if first not in READ_ONLY_TERMINAL_COMMANDS:
            return False
    return True


def default_security_risk(tool_name: str, args: dict[str, Any]) -> str:
    if tool_name == "terminal":
        return "LOW" if terminal_command_is_read_only(args.get("command")) else "MEDIUM"
    if tool_name == "file_editor":
        key = f"file_editor:{args.get('command')}"
        return "LOW" if key in READ_ONLY_TOOLS else "MEDIUM"
    if tool_name == "task_tracker":
        key = f"task_tracker:{args.get('command')}"
        return "LOW" if key in READ_ONLY_TOOLS else "MEDIUM"
    if tool_name in READ_ONLY_TOOLS:
        return "LOW"
    if tool_name.startswith("browser_"):
        return "LOW" if tool_name in READ_ONLY_TOOLS else "MEDIUM"
    return "UNKNOWN"


def summary_from_content(content: str, tool_name: str) -> str:
    text = " ".join(content.split())
    if text:
        return text[:120]
    return f"Call {tool_name}"


def make_tool_call(
    index: int,
    tool_name: str,
    args: dict[str, Any],
    thought: str,
    explicit_id: str | None = None,
) -> dict[str, Any]:
    enriched_args = dict(args)
    if supports_security_risk(tool_name):
        enriched_args.setdefault("security_risk", default_security_risk(tool_name, args))
    enriched_args.setdefault("summary", summary_from_content(thought, tool_name))
    return {
        "id": explicit_id or tool_call_id(index, tool_name),
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(enriched_args, ensure_ascii=False),
        },
    }


def web_observation_to_content(event: WebObservation) -> str:
    chunks = []
    if event.url:
        chunks.append(f"URL: {event.url}")
    if event.axtree:
        chunks.append(event.axtree)
    elif event.html:
        chunks.append(event.html)
    return "\n\n".join(chunks)


class ConversionState:
    def __init__(
        self,
        dataset_tools: dict[str, dict[str, Any]],
        system_prompt: str,
        trajectory: Trajectory,
        metadata_code_enabled: list[str],
        metadata_browser_enabled: bool,
    ):
        self.dataset_tools = dataset_tools
        self.system_prompt = system_prompt
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": text_content(system_prompt)}
        ]
        self.tools_by_name: dict[str, dict[str, Any]] = {}
        self.pending_tool_calls: list[dict[str, str]] = []
        self.call_index = 0
        self.metadata_browser_enabled = metadata_browser_enabled
        self.enable_metadata_tools(
            trajectory,
            metadata_code_enabled=metadata_code_enabled,
            metadata_browser_enabled=metadata_browser_enabled,
        )

    def add_builtin_tool(self, name: str) -> None:
        self.tools_by_name[name] = BUILTIN_TOOLS[name]

    def enable_metadata_tools(
        self,
        trajectory: Trajectory,
        *,
        metadata_code_enabled: list[str],
        metadata_browser_enabled: bool,
    ) -> None:
        unsupported_code_languages = sorted(set(metadata_code_enabled) - {"bash"})
        if unsupported_code_languages:
            raise ValueError(
                "OpenHands SDK SFT conversion only supports bash code actions. "
                f"Unsupported code_enabled languages: {unsupported_code_languages}"
            )
        if "bash" in metadata_code_enabled:
            self.add_builtin_tool("terminal")
        if metadata_browser_enabled:
            for tool_name in BROWSER_BUILTIN_TOOL_NAMES:
                self.add_builtin_tool(tool_name)

        available_custom_tools = (
            trajectory.available_custom_tools
            if trajectory.available_custom_tools is not None
            else sorted(self.dataset_tools)
        )
        for source_tool_name in available_custom_tools:
            if metadata_browser_enabled and source_tool_name in BROWSER_TOOL_ALIASES:
                continue
            tool_name = OPENHANDS_TOOL_ALIASES.get(source_tool_name, source_tool_name)
            if tool_name in BUILTIN_TOOLS:
                self.add_builtin_tool(tool_name)
            elif source_tool_name in self.dataset_tools:
                self.tools_by_name[source_tool_name] = self.dataset_tools[source_tool_name]
            else:
                raise ValueError(
                    f"available_custom_tools contains {source_tool_name!r}, "
                    "but metadata.json does not define that custom tool"
                )

    def add_tool(self, name: str, args: dict[str, Any]) -> None:
        if name in self.tools_by_name:
            return
        if name in BUILTIN_TOOLS:
            self.tools_by_name[name] = BUILTIN_TOOLS[name]
        elif name in self.dataset_tools:
            self.tools_by_name[name] = self.dataset_tools[name]
        else:
            raise ValueError(f"Tool {name!r} is not declared in metadata.json")

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": text_content(content)})

    def add_assistant_message(
        self,
        content: str,
        reasoning_content: str | None = None,
    ) -> None:
        message: dict[str, Any] = {"role": "assistant"}
        if content:
            message["content"] = text_content(content)
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
        if "content" not in message and "reasoning_content" not in message:
            message["content"] = text_content("")
        self.messages.append(message)

    def add_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        content: str,
        reasoning_content: str | None = None,
        explicit_id: str | None = None,
    ) -> None:
        self.add_tool(tool_name, args)
        self.call_index += 1
        summary_source = "\n\n".join(part for part in [content, reasoning_content] if part)
        tool_call = make_tool_call(
            self.call_index,
            tool_name,
            args,
            summary_source,
            explicit_id=explicit_id,
        )
        pending = {
            "id": tool_call["id"],
            "name": tool_name,
            "fallback": self.synthetic_tool_result(tool_name, args),
        }
        if (
            self.messages
            and self.messages[-1].get("role") == "assistant"
            and "tool_calls" in self.messages[-1]
            and self.pending_tool_calls
        ):
            if content:
                existing_content = self.message_text(self.messages[-1])
                merged = f"{existing_content}\n\n{content}" if existing_content else content
                self.messages[-1]["content"] = text_content(merged)
            if reasoning_content:
                existing_reasoning = self.messages[-1].get("reasoning_content") or ""
                self.messages[-1]["reasoning_content"] = (
                    f"{existing_reasoning}\n\n{reasoning_content}"
                    if existing_reasoning
                    else reasoning_content
                )
            self.messages[-1]["tool_calls"].append(tool_call)
        else:
            message = {"role": "assistant", "tool_calls": [tool_call]}
            if content:
                message["content"] = text_content(content)
            if reasoning_content:
                message["reasoning_content"] = reasoning_content
            self.messages.append(message)
        self.pending_tool_calls.append(pending)

    def add_tool_result(self, content: str, explicit_id: str | None = None) -> None:
        if not self.pending_tool_calls:
            self.add_user_message(content)
            return
        if explicit_id is None:
            pending_index = 0
        else:
            pending_index = next(
                (
                    index
                    for index, pending_tool_call in enumerate(self.pending_tool_calls)
                    if pending_tool_call["id"] == explicit_id
                ),
                None,
            )
            if pending_index is None:
                raise ValueError(f"No pending tool call found for id {explicit_id!r}")
        pending = self.pending_tool_calls.pop(pending_index)
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": pending["id"],
                "name": pending["name"],
                "content": text_content(content),
            }
        )

    def close_pending_tools(self) -> None:
        while self.pending_tool_calls:
            pending = self.pending_tool_calls.pop(0)
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": pending["id"],
                    "name": pending["name"],
                    "content": text_content(pending["fallback"]),
                }
            )

    @staticmethod
    def synthetic_tool_result(tool_name: str, args: dict[str, Any]) -> str:
        if tool_name == "finish":
            return str(args.get("message", ""))
        if tool_name == "think":
            return "Your thought has been logged."
        return ""

    def sorted_tools(self) -> list[dict[str, Any]]:
        return [self.tools_by_name[name] for name in sorted(self.tools_by_name)]

    def api_action_is_browser(self, event: ApiAction) -> bool:
        return is_browser_api_action(
            event.function,
            event.kwargs,
            browser_context=self.metadata_browser_enabled,
        )

    @staticmethod
    def message_text(message: dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return ""


def convert_event(state: ConversionState, event: Any) -> None:
    if isinstance(event, TextObservation):
        if state.pending_tool_calls:
            state.add_tool_result(event.content, event.tool_call_id)
        elif event.source == "agent":
            state.add_assistant_message(event.content)
        else:
            state.add_user_message(event.content)
        return

    if isinstance(event, WebObservation):
        content = web_observation_to_content(event)
        if state.pending_tool_calls:
            state.add_tool_result(content, event.tool_call_id)
        else:
            state.add_user_message(content)
        return

    if isinstance(event, ImageObservation):
        content = f"[Image: {event.content}]"
        if event.annotations:
            annotations = [
                f"{annotation.text} ({annotation.element_type})"
                for annotation in event.annotations
                if annotation.text
            ]
            if annotations:
                content += "\nElements detected: " + ", ".join(annotations)
        if state.pending_tool_calls:
            state.add_tool_result(content, event.tool_call_id)
        else:
            state.add_user_message(content)
        return

    if isinstance(event, ApiAction):
        tool_name, args = map_api_action(event, browser_action=state.api_action_is_browser(event))
        state.add_tool_call(
            tool_name,
            args,
            action_content(event),
            event_reasoning_content(event),
            explicit_id=event.tool_call_id,
        )
        return

    if isinstance(event, CodeAction):
        tool_name, args = map_code_action(event)
        state.add_tool_call(
            tool_name,
            args,
            action_content(event),
            event_reasoning_content(event),
            explicit_id=event.tool_call_id,
        )
        return

    if isinstance(event, MessageAction):
        finish_message = extract_finish_message(event.content)
        if finish_message is not None:
            state.add_tool_call(
                "finish",
                {"message": finish_message},
                action_content(event),
                event_reasoning_content(event),
                explicit_id=event.tool_call_id,
            )
        elif legacy_tool_call := extract_legacy_tool_call(event.content):
            function_name, kwargs, thought = legacy_tool_call
            api_action = ApiAction(function=function_name, kwargs=kwargs)
            tool_name, args = map_api_action(
                api_action,
                browser_action=state.api_action_is_browser(api_action),
            )
            content = "\n\n".join(part for part in [action_content(event), thought] if part)
            state.add_tool_call(
                tool_name,
                args,
                content,
                event_reasoning_content(event),
                explicit_id=event.tool_call_id,
            )
        else:
            state.close_pending_tools()
            state.add_assistant_message(
                "\n\n".join(part for part in [action_content(event), event.content] if part),
                event_reasoning_content(event),
            )
        return

    raise ValueError(f"Unsupported event type: {type(event)}")


def process_row(line: str, system_prompt: str) -> dict[str, Any]:
    trajectory = Trajectory(**json.loads(line))
    metadata = load_dataset_metadata(dataset, required=bool(dataset))
    validate_trajectory_metadata(
        trajectory,
        metadata,
        dataset_name=dataset,
        native_api_names=OPENHANDS_SDK_NATIVE_API_NAMES,
    )
    state = ConversionState(
        custom_tool_map(metadata),
        system_prompt,
        trajectory,
        metadata_code_enabled=metadata.code_enabled,
        metadata_browser_enabled=metadata.browser_enabled,
    )
    for event in trajectory.content:
        convert_event(state, event)
    state.close_pending_tools()
    return {
        "id": trajectory.id,
        "messages": state.messages,
        "tools": state.sorted_tools(),
        "metadata": {
            "agent": "openhands_sdk",
            "format": "openai_chat_completions",
            "source_dataset": dataset,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert standardized data to OpenHands SDK V1 SFT format"
    )
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="Optional file containing the system prompt to place in each record.",
    )
    # Accepted for compatibility with scripts that invoke openhands_v0.
    parser.add_argument("--is_web", choices=["yes", "no"], default=None)
    parser.add_argument("--api_env", default=None)
    args = parser.parse_args()

    system_prompt = load_system_prompt(args.system_prompt_file)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        print(json.dumps(process_row(line, system_prompt), ensure_ascii=False))


if __name__ == "__main__":
    main()
