from __future__ import annotations

# ruff: noqa: I001
import json
import os
import platform
import re
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

from pydantic import Field, SecretStr

from openhands.sdk import (
    Action,
    Agent,
    Conversation,
    Event,
    ImageContent,
    LLM,
    Observation,
    TextContent,
    ToolDefinition,
    Workspace,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import ActionEvent, ObservationEvent
from openhands.sdk.tool import Tool, ToolExecutor, register_tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

try:
    from openhands.tools.browser_use import BrowserToolSet
except Exception:  # noqa: BLE001
    BrowserToolSet = None

try:
    from openhands.workspace import DockerWorkspace
except Exception:  # noqa: BLE001
    DockerWorkspace = None


DATASET_NAME = "CharlieDreemur_OpenManus-RL"
RECORD_INDEX = 0
RECORD_ID = "look_at_obj_in_light-AlarmClock-None-DeskLamp-301_trial_T20190907_174127_043461"
MODEL = os.getenv("LLM_MODEL", "openhands/minimax-m2.7")
MAX_VALIDATION_ITERATIONS = int(os.getenv("MAX_VALIDATION_ITERATIONS", "8"))
BROWSER_BUILTIN_TOOL_NAMES = {
    "browser_click",
    "browser_get_content",
    "browser_get_state",
    "browser_go_back",
    "browser_navigate",
    "browser_scroll",
    "browser_type",
}
BROWSER_TOOL_ALIASES = {
    "back": "browser_go_back",
    "click": "browser_click",
    "fill": "browser_type",
    "go_back": "browser_go_back",
    "goto": "browser_navigate",
    "scroll": "browser_scroll",
    "type": "browser_type",
}
OPENHANDS_TOOL_ALIASES = {
    "edit_file": "file_editor",
    "finish": "finish",
    "str_replace_editor": "file_editor",
    "stop": "finish",
    "submit": "finish",
    "task_tracker": "task_tracker",
    "think": "think",
}
SDK_NATIVE_TOOL_NAMES = {
    "file_editor",
    "finish",
    "task_tracker",
    "terminal",
    "think",
    *BROWSER_BUILTIN_TOOL_NAMES,
}


class MetadataToolObservation(Observation):
    tool_name: str = Field(description="Name of the metadata custom tool that ran.")
    arguments: dict[str, Any] = Field(default_factory=dict)

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        args = json.dumps(self.arguments, ensure_ascii=False, sort_keys=True)
        return [
            TextContent(
                text=(
                    f"The metadata custom tool {self.tool_name!r} executed with arguments: {args}."
                )
            )
        ]


class MetadataToolExecutor(ToolExecutor):
    def __init__(self, tool_name: str):
        self.tool_name = tool_name

    def __call__(
        self,
        action: Action,
        conversation: Conversation | None = None,  # noqa: ARG002
    ) -> MetadataToolObservation:
        return MetadataToolObservation(
            tool_name=self.tool_name,
            arguments=action.model_dump(mode="json"),
        )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def output_dir() -> Path:
    return Path(__file__).resolve().parent


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def class_name(name: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", name)
    text = "".join(part[:1].upper() + part[1:] for part in parts if part)
    if not text or text[0].isdigit():
        text = "Dataset" + text
    return text


def load_dataset_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    root = repo_root()
    metadata_path = root / "datasets" / DATASET_NAME / "metadata.json"
    sample_path = root / "datasets" / DATASET_NAME / "sample_std.json"
    metadata = json.loads(metadata_path.read_text())
    rows = json.loads(sample_path.read_text())
    row = rows[RECORD_INDEX]
    if row.get("id") != RECORD_ID:
        raise RuntimeError(f"Expected {RECORD_ID!r} at index {RECORD_INDEX}, got {row.get('id')!r}")
    return metadata, row


def first_task_message(row: dict[str, Any]) -> str:
    for event in row.get("content", []):
        if (
            event.get("class_") == "text_observation"
            and event.get("source") == "user"
            and event.get("content")
        ):
            return str(event["content"])
    for event in row.get("content", []):
        if event.get("class_") == "message_action" and event.get("content"):
            return str(event["content"])
    raise RuntimeError("First standardized record does not contain a user task message")


def metadata_custom_tool_specs(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    specs = {}
    for tool in metadata.get("custom_tools") or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name"):
            specs[str(function["name"])] = function
    return specs


def available_custom_tool_names(metadata: dict[str, Any], row: dict[str, Any]) -> list[str]:
    tool_specs = metadata_custom_tool_specs(metadata)
    names = row.get("available_custom_tools")
    if names is None:
        return sorted(tool_specs)
    return [str(name) for name in names]


def normalize_parameters(function: dict[str, Any]) -> dict[str, Any]:
    parameters = function.get("parameters")
    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}}
    if parameters.get("type") != "object":
        parameters = {**parameters, "type": "object"}
    parameters.setdefault("properties", {})
    return parameters


def make_metadata_tool(name: str, function: dict[str, Any]) -> type[ToolDefinition]:
    parameters = normalize_parameters(function)
    action_type = Action.from_mcp_schema(f"{class_name(name)}Action", parameters)
    description = function.get("description") or f"Dataset metadata tool {name}."

    def create(
        cls,
        conv_state=None,  # noqa: ARG001
        _name=name,
        _description=description,
        _action_type=action_type,
        **params,  # noqa: ARG001
    ) -> list[ToolDefinition]:
        return [
            cls(
                description=_description,
                action_type=_action_type,
                observation_type=MetadataToolObservation,
                executor=MetadataToolExecutor(_name),
            )
        ]

    return type(
        f"{class_name(name)}Tool",
        (ToolDefinition,),
        {"name": name, "create": classmethod(create)},
    )


def exact_tool_filter(tool_names: set[str]) -> str | None:
    if not tool_names:
        return None
    return "^(?:" + "|".join(re.escape(name) for name in sorted(tool_names)) + ")$"


def register_metadata_tools(
    metadata: dict[str, Any], row: dict[str, Any]
) -> tuple[list[Tool], dict[str, Any]]:
    tool_specs = metadata_custom_tool_specs(metadata)
    available_names = available_custom_tool_names(metadata, row)
    code_enabled = set(metadata.get("code_enabled") or [])
    browser_enabled = bool(metadata.get("browser_enabled"))
    tools: list[Tool] = []
    filtered_tool_names: set[str] = set()
    custom_tool_names: list[str] = []
    builtin_sources: dict[str, list[str]] = {}
    include_default_tools = ["FinishTool"]

    def add_tool_name(name: str, source: str) -> None:
        filtered_tool_names.add(name)
        builtin_sources.setdefault(name, []).append(source)

    if "bash" in code_enabled:
        tools.append(Tool(name=TerminalTool.name))
        add_tool_name(TerminalTool.name, "metadata.code_enabled:bash")

    if browser_enabled:
        if BrowserToolSet is None:
            raise RuntimeError(
                "metadata.browser_enabled is true, but BrowserToolSet is unavailable"
            )
        tools.append(Tool(name=BrowserToolSet.name))
        for name in BROWSER_BUILTIN_TOOL_NAMES:
            add_tool_name(name, "metadata.browser_enabled:true")

    for source_name in available_names:
        mapped_name = OPENHANDS_TOOL_ALIASES.get(source_name, source_name)
        if browser_enabled and source_name in BROWSER_TOOL_ALIASES:
            continue
        if mapped_name == "terminal":
            if "bash" not in code_enabled:
                raise RuntimeError(
                    f"{source_name!r} maps to terminal, but metadata.code_enabled "
                    "does not include 'bash'"
                )
            continue
        if mapped_name == "file_editor":
            if FileEditorTool.name not in filtered_tool_names:
                tools.append(Tool(name=FileEditorTool.name))
            add_tool_name(FileEditorTool.name, f"available_custom_tools:{source_name}")
            continue
        if mapped_name == "task_tracker":
            if TaskTrackerTool.name not in filtered_tool_names:
                tools.append(Tool(name=TaskTrackerTool.name))
            add_tool_name(TaskTrackerTool.name, f"available_custom_tools:{source_name}")
            continue
        if mapped_name == "think":
            if "ThinkTool" not in include_default_tools:
                include_default_tools.append("ThinkTool")
            continue
        if mapped_name == "finish":
            continue
        if source_name not in tool_specs:
            raise RuntimeError(
                f"available_custom_tools contains {source_name!r}, but metadata.json "
                "does not define that custom tool"
            )
        register_tool(source_name, make_metadata_tool(source_name, tool_specs[source_name]))
        tools.append(Tool(name=source_name))
        add_tool_name(source_name, f"available_custom_tools:{source_name}")
        custom_tool_names.append(source_name)

    return tools, {
        "available_custom_tools": available_names,
        "registered_custom_tools": custom_tool_names,
        "filtered_tool_names": sorted(filtered_tool_names),
        "filter_tools_regex": exact_tool_filter(filtered_tool_names),
        "include_default_tools": include_default_tools,
        "builtin_tool_sources": builtin_sources,
    }


def detect_platform() -> str:
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "linux/arm64"
    return "linux/amd64"


def server_image() -> str:
    platform_str = detect_platform()
    arch = "arm64" if "arm64" in platform_str else "amd64"
    sha = os.getenv("SDK_SHA") or os.getenv("GITHUB_SHA")
    if sha:
        return f"ghcr.io/openhands/agent-server:{sha[:7]}-python-{arch}"
    return os.getenv("OPENHANDS_AGENT_SERVER_IMAGE", "ghcr.io/openhands/agent-server:latest-python")


@contextmanager
def validation_workspace() -> Iterator[tuple[Any, str, str | None]]:
    mode = os.getenv("VALIDATION_WORKSPACE", "docker").lower()
    if mode == "local":
        with tempfile.TemporaryDirectory(prefix=f"{DATASET_NAME}-workspace-") as tmpdir:
            yield Workspace(working_dir=tmpdir), "local", None
        return
    if mode == "remote":
        host = os.getenv("VALIDATION_WORKSPACE_HOST")
        if not host:
            raise RuntimeError("VALIDATION_WORKSPACE_HOST is required for remote mode")
        with tempfile.TemporaryDirectory(prefix=f"{DATASET_NAME}-workspace-") as tmpdir:
            workspace = Workspace(host=host, working_dir=tmpdir)
            yield workspace, "remote", None
        return
    if mode != "docker":
        raise RuntimeError(f"Unsupported VALIDATION_WORKSPACE mode: {mode}")
    if DockerWorkspace is None:
        raise RuntimeError("DockerWorkspace is unavailable")
    try:
        with DockerWorkspace(server_image=server_image(), platform=detect_platform()) as workspace:
            yield workspace, "docker", None
    except Exception as exc:  # noqa: BLE001
        if os.getenv("VALIDATION_ALLOW_LOCAL_FALLBACK") != "1":
            raise
        with tempfile.TemporaryDirectory(prefix=f"{DATASET_NAME}-workspace-") as tmpdir:
            yield (
                Workspace(working_dir=tmpdir),
                "local_fallback",
                f"{exc.__class__.__name__}: {exc}",
            )


def latest_log(log_dir: Path) -> Path | None:
    logs = sorted(log_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
    if not logs:
        return None
    return logs[-1]


def event_summary(events: list[Event]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    actions: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for event in events:
        if isinstance(event, ActionEvent):
            actions.append(
                {
                    "id": str(event.id),
                    "tool_name": event.tool_name,
                    "tool_call_id": event.tool_call_id,
                    "action": event.action.model_dump(mode="json")
                    if event.action is not None
                    else None,
                }
            )
        elif isinstance(event, ObservationEvent):
            observations.append(
                {
                    "action_id": str(event.action_id),
                    "tool_name": event.tool_name,
                    "tool_call_id": event.tool_call_id,
                    "observation": event.observation.model_dump(mode="json"),
                }
            )
    return actions, observations


def write_completion(log_dir: Path, error: str | None) -> bool:
    completion_path = output_dir() / "completion.json"
    latest = latest_log(log_dir)
    if latest is None:
        completion_path.write_text(
            json.dumps(
                {
                    "dataset": DATASET_NAME,
                    "completion_written": False,
                    "error": error or "No LLM completion log was written.",
                },
                indent=2,
            )
            + "\n"
        )
        return False
    completion_path.write_text(json.dumps(json.loads(latest.read_text()), indent=2) + "\n")
    return True


def run_validation() -> None:
    root = repo_root()
    load_env_file(root / ".env")
    load_env_file(Path.home() / "work" / "agent-data-protocol" / ".env")
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is required")

    metadata, row = load_dataset_inputs()
    user_message = first_task_message(row)
    sdk_tools, tool_info = register_metadata_tools(metadata, row)
    log_dir = Path(tempfile.mkdtemp(prefix=f"{DATASET_NAME}-agent-completions-"))
    events: list[Event] = []
    start_time = time.time()

    def collect_event(event: Event) -> None:
        events.append(event)

    llm = LLM(
        usage_id="validation-agent",
        model=MODEL,
        api_key=SecretStr(api_key),
        base_url=os.getenv("LLM_BASE_URL"),
        log_completions=True,
        log_completions_folder=str(log_dir),
        max_output_tokens=int(os.getenv("MAX_VALIDATION_OUTPUT_TOKENS", "2048")),
    )
    agent = Agent(
        llm=llm,
        tools=sdk_tools,
        filter_tools_regex=tool_info["filter_tools_regex"],
        include_default_tools=tool_info["include_default_tools"],
    )

    final_status = ConversationExecutionStatus.ERROR
    workspace_mode = None
    workspace_note = None
    error = None
    conversation = None
    try:
        with validation_workspace() as (workspace, workspace_mode, workspace_note):
            conversation = Conversation(
                agent=agent,
                workspace=workspace,
                callbacks=[collect_event],
                visualizer=None,
                max_iteration_per_run=MAX_VALIDATION_ITERATIONS,
                stuck_detection=False,
            )
            conversation.send_message(user_message)
            conversation.run()
            final_status = conversation.state.execution_status
    except Exception as exc:  # noqa: BLE001
        error = f"{exc.__class__.__name__}: {exc}"
    finally:
        if conversation is not None:
            try:
                conversation.close()
            except Exception:  # noqa: BLE001
                pass

    completion_written = write_completion(log_dir, error)
    actions, observations = event_summary(events)
    action_names = [action["tool_name"] for action in actions]
    run_path = output_dir() / "run.json"
    run_path.write_text(
        json.dumps(
            {
                "dataset": DATASET_NAME,
                "record_id": row.get("id"),
                "record_index": RECORD_INDEX,
                "model": MODEL,
                "selected_user_text": user_message,
                "validation_mode": (
                    "OpenHands SDK Agent/Conversation with metadata-derived tools "
                    "and a real SDK workspace"
                ),
                "workspace_mode": workspace_mode,
                "workspace_note": workspace_note,
                "code_enabled": metadata.get("code_enabled") or [],
                "browser_enabled": bool(metadata.get("browser_enabled")),
                **tool_info,
                "max_iterations": MAX_VALIDATION_ITERATIONS,
                "final_status": str(final_status),
                "ran_to_finished_state": final_status == ConversationExecutionStatus.FINISHED,
                "performed_tool_call": any(action.get("action") is not None for action in actions),
                "action_names": action_names,
                "actions": actions,
                "observations": observations,
                "completion_written": completion_written,
                "elapsed_seconds": round(time.time() - start_time, 3),
                "error": error,
            },
            indent=2,
        )
        + "\n"
    )
    if error:
        raise RuntimeError(error)


def main() -> None:
    run_validation()


if __name__ == "__main__":
    main()
