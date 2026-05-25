from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    Event,
    ImageContent,
    Message,
    Observation,
    TextContent,
    Tool,
    ToolDefinition,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import ActionEvent, ObservationEvent
from openhands.sdk.tool import Action, ToolExecutor, register_tool
from pydantic import SecretStr

MODEL = os.getenv("LLM_MODEL", "openhands/minimax-m2.7")
DEFAULT_MAX_ITERATIONS = int(os.getenv("MAX_VALIDATION_ITERATIONS", "30"))
MAX_VALIDATION_ITERATIONS_CAP = int(os.getenv("MAX_VALIDATION_ITERATIONS_CAP", "80"))
BUILTIN_TOOL_NAMES = {"finish", "think"}
NON_ENVIRONMENT_TOOL_NAMES = {"finish", "think"}


class ReplayObservation(Observation):
    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if self.content:
            return self.content
        return [TextContent(text="The mocked validation tool completed.")]


class ReplayState:
    def __init__(self, observations_by_tool: dict[str, list[str]]):
        self.observations_by_tool = {
            name: list(values) for name, values in observations_by_tool.items()
        }
        self.indices = {name: 0 for name in observations_by_tool}
        self.extra_calls = 0

    def result_for(self, tool_name: str) -> str:
        observations = self.observations_by_tool.get(tool_name, [])
        index = self.indices.get(tool_name, 0)
        self.indices[tool_name] = index + 1
        if index < len(observations):
            result = observations[index]
            if self.all_recorded_observations_consumed():
                return result + self.done_note()
            return result + self.continue_note()
        self.extra_calls += 1
        return (
            f"The mocked {tool_name} action completed successfully. The mocked "
            "validation environment has no further recorded observations for this "
            "sample and reports that the task goal is satisfied; call finish now."
        )

    def all_recorded_observations_consumed(self) -> bool:
        return all(
            self.indices.get(name, 0) >= len(observations)
            for name, observations in self.observations_by_tool.items()
        )

    @staticmethod
    def continue_note() -> str:
        return (
            "\n\nValidation note: this is a recorded intermediate observation from "
            "the dataset trajectory. Continue solving the task with the available "
            "tools; do not call finish until the task goal is satisfied."
        )

    @staticmethod
    def done_note() -> str:
        return (
            "\n\nValidation note: the mocked validation environment reports that "
            "this tool action completed the task goal. This is the authoritative "
            "environment state and overrides any earlier instruction to keep "
            "exploring, editing, or testing. Your next action must be the finish "
            "tool; do not call any other tool."
        )


class ReplayExecutor(ToolExecutor):
    def __init__(self, tool_name: str, replay_state: ReplayState):
        self.tool_name = tool_name
        self.replay_state = replay_state

    def __call__(
        self, action: Action, conversation: Conversation | None = None
    ) -> ReplayObservation:
        return ReplayObservation.from_text(self.replay_state.result_for(self.tool_name))


class ReplayTool(ToolDefinition):
    @classmethod
    def create(cls, *args, **kwargs) -> list[Self]:
        return []


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    texts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            texts.append(item.get("text", ""))
    return "\n".join(text for text in texts if text)


def sdk_content(content: Any) -> list[TextContent | ImageContent]:
    if content is None:
        return []
    if isinstance(content, str):
        return [TextContent(text=content)]
    blocks: list[TextContent | ImageContent] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            blocks.append(TextContent(text=item.get("text", "")))
        elif item.get("type") in {"image", "image_url"}:
            image_url = item.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if url:
                blocks.append(ImageContent(image_urls=[url]))
    return blocks


def looks_like_observation(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith(
        (
            "URL:",
            "RootWebArea",
            "[Image:",
            "Elements detected:",
            "<html",
            "We need perform a task",
        )
    )


def context_message(record: dict[str, Any]) -> tuple[list[int], Message]:
    messages = record["messages"]
    first_tool_index = next(
        (
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant" and message.get("tool_calls")
        ),
        None,
    )
    context_messages = []
    if first_tool_index is not None:
        seen_user = False
        for index, message in enumerate(messages[:first_tool_index]):
            role = message.get("role")
            if role == "user" and sdk_content(message.get("content")):
                seen_user = True
                context_messages.append((index, message))
            elif role == "assistant" and not seen_user and sdk_content(message.get("content")):
                context_messages.append((index, message))
    if not context_messages:
        context_messages = [
            (index, message)
            for index, message in enumerate(messages)
            if message.get("role") in {"assistant", "user"}
            and not message.get("tool_calls")
            and sdk_content(message.get("content"))
        ][:1]

    content: list[TextContent | ImageContent] = []
    indices: list[int] = []
    expects_environment = environment_tool_call_count(record) > 0
    if expects_environment:
        content.append(
            TextContent(
                text=(
                    "Validation harness instruction: this is a live OpenHands SDK "
                    "format check using mocked dataset tools. Work on the real "
                    "dataset task below, but make at least one environment tool "
                    "call before finishing. When a mocked tool result reports that "
                    "the task goal is satisfied, stop immediately and call finish "
                    "as the next action. Do not answer only in plain text, and do "
                    "not continue using tools after a mocked result says the task "
                    "is satisfied. Use only the declared SDK tools for this run; "
                    "do not emit legacy XML tool syntax or call undeclared tools "
                    "such as bash, grep, or shell unless those names appear in the "
                    "available SDK tool list.\n---\n"
                )
            )
        )
    for index, message in context_messages:
        indices.append(index)
        role = message.get("role", "message")
        text = text_from_content(message.get("content"))
        if content:
            content.append(TextContent(text="\n---\n"))
        if role == "assistant":
            content.append(TextContent(text="Task:"))
        elif looks_like_observation(text):
            content.append(TextContent(text="Current environment observation:"))
        elif content:
            content.append(TextContent(text="Additional user context:"))
        content.extend(sdk_content(message.get("content")))
    if content:
        if expects_environment:
            content.append(
                TextContent(
                    text=(
                        "\n\nUse the available tools to interact with the "
                        "environment and make progress on the task. You must "
                        "make at least one environment tool call before finishing; "
                        "do not answer only in plain text. When the mocked tool "
                        "result reports that the task goal is satisfied, call "
                        "finish immediately as the next action. Use only declared "
                        "SDK tools; do not emit legacy XML tool syntax or call "
                        "undeclared shell tools."
                    )
                )
            )
        return indices, Message(role="user", content=content)
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            return [index], Message(role="user", content=sdk_content(message.get("content")))
    raise RuntimeError("Selected record does not contain a user message")


def tool_call_names(record: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for message in record["messages"]:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            name = tool_call["function"]["name"]
            if name not in names:
                names.append(name)
    return names


def tool_call_count(record: dict[str, Any]) -> int:
    return sum(
        len(message.get("tool_calls") or [])
        for message in record["messages"]
        if message.get("role") == "assistant"
    )


def environment_tool_call_count(record: dict[str, Any]) -> int:
    return sum(
        1
        for message in record["messages"]
        if message.get("role") == "assistant"
        for tool_call in message.get("tool_calls") or []
        if tool_call["function"]["name"] not in NON_ENVIRONMENT_TOOL_NAMES
    )


def observations_by_tool(record: dict[str, Any]) -> dict[str, list[str]]:
    pending: dict[str, str] = {}
    observations: dict[str, list[str]] = {}
    for message in record["messages"]:
        if message.get("role") == "assistant":
            for tool_call in message.get("tool_calls") or []:
                pending[tool_call["id"]] = tool_call["function"]["name"]
        elif message.get("role") == "tool":
            tool_call_id = message.get("tool_call_id")
            tool_name = pending.pop(tool_call_id, "") if tool_call_id else ""
            if not tool_name:
                continue
            observations.setdefault(tool_name, []).append(text_from_content(message.get("content")))
    return observations


def tool_specs_by_name(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        tool["function"]["name"]: tool["function"]
        for tool in record.get("tools", [])
        if tool.get("type") == "function" and tool.get("function", {}).get("name")
    }


def make_replay_tool(
    name: str, function: dict[str, Any], replay_state: ReplayState
) -> type[ToolDefinition]:
    parameters = function.get("parameters") or {"type": "object", "properties": {}}
    action_type = Action.from_mcp_schema(f"{class_name(name)}Action", parameters)

    def create(
        cls,
        conv_state=None,  # noqa: ARG001
        _name=name,
        _function=function,
        _action_type=action_type,
        _replay_state=replay_state,
        **params,  # noqa: ARG001
    ) -> list[Self]:
        return [
            cls(
                description=_function.get("description") or f"Mocked validation tool {_name}.",
                action_type=_action_type,
                observation_type=ReplayObservation,
                executor=ReplayExecutor(_name, _replay_state),
            )
        ]

    return type(
        f"{class_name(name)}ReplayTool",
        (ReplayTool,),
        {"name": name, "create": classmethod(create)},
    )


def register_replay_tools(record: dict[str, Any]) -> tuple[list[Tool], list[str]]:
    replay_state = ReplayState(observations_by_tool(record))
    specs = tool_specs_by_name(record)
    tools: list[Tool] = []
    mocked: list[str] = []
    for name in tool_call_names(record):
        if name in BUILTIN_TOOL_NAMES:
            continue
        function = specs.get(
            name,
            {
                "name": name,
                "description": f"Mocked validation tool {name}.",
                "parameters": {"type": "object", "properties": {}},
            },
        )
        register_tool(name, make_replay_tool(name, function, replay_state))
        tools.append(Tool(name=name))
        mocked.append(name)
    return tools, mocked


def latest_log(log_dir: Path) -> Path | None:
    logs = sorted(log_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
    if not logs:
        return None
    return logs[-1]


def event_summary(
    events: list[Event],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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


def run_dataset_validation(dataset_name: str, record: dict[str, Any]) -> None:
    root = repo_root()
    load_env_file(root / ".env")
    load_env_file(Path.home() / "work" / "agent-data-protocol" / ".env")
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is required")

    selected_message_indices, task_message = context_message(record)
    log_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_name}-agent-completions-"))
    workspace = Path(tempfile.mkdtemp(prefix=f"{dataset_name}-workspace-"))
    output_dir = Path(__file__).resolve().parent / dataset_name
    completion_path = output_dir / "completion.json"
    run_path = output_dir / "run.json"

    tools, mocked_tools = register_replay_tools(record)
    expected_tool_calls = tool_call_count(record)
    expected_environment_tool_calls = environment_tool_call_count(record)
    max_iterations = min(
        MAX_VALIDATION_ITERATIONS_CAP,
        max(DEFAULT_MAX_ITERATIONS, expected_tool_calls + 5),
    )

    llm = LLM(
        model=MODEL,
        api_key=SecretStr(api_key),
        base_url=os.getenv("LLM_BASE_URL"),
        log_completions=True,
        log_completions_folder=str(log_dir),
        max_output_tokens=2048,
    )
    agent = Agent(
        llm=llm,
        tools=tools,
        include_default_tools=["FinishTool", "ThinkTool"],
    )
    events: list[Event] = []

    def collect_event(event: Event) -> None:
        events.append(event)

    completion_written = False
    error: str | None = None
    try:
        conversation = Conversation(
            agent=agent,
            callbacks=[collect_event],
            workspace=workspace,
            visualizer=None,
            max_iteration_per_run=max_iterations,
            stuck_detection=False,
        )
        conversation.send_message(task_message)
        conversation.run()
        final_status = conversation.state.execution_status
    except Exception as exc:  # noqa: BLE001
        final_status = ConversationExecutionStatus.ERROR
        error = f"{exc.__class__.__name__}: {exc}"
    finally:
        latest = latest_log(log_dir)
        if latest is not None:
            completion_path.write_text(json.dumps(json.loads(latest.read_text()), indent=2) + "\n")
            completion_written = True

    actions, observations = event_summary(events)
    action_names = [action["tool_name"] for action in actions]
    valid_actions = [action for action in actions if action["action"] is not None]
    environment_actions = [
        action for action in valid_actions if action["tool_name"] not in NON_ENVIRONMENT_TOOL_NAMES
    ]
    called_finish_tool = any(action["tool_name"] == "finish" for action in valid_actions)
    reached_finish = final_status == ConversationExecutionStatus.FINISHED or called_finish_tool
    performed_tool_call = bool(valid_actions)
    performed_environment_tool_call = bool(environment_actions)
    completed_with_tool_execution = reached_finish and performed_environment_tool_call
    completed_with_any_tool_call = reached_finish and performed_tool_call
    expects_environment_tool_calls = expected_environment_tool_calls > 0
    completed_as_expected = reached_finish and (
        completed_with_tool_execution or not expects_environment_tool_calls
    )
    validation_status = "completed" if completed_as_expected else "incomplete"
    failure_reason = None
    if not completed_as_expected:
        if not reached_finish and not expects_environment_tool_calls:
            failure_reason = (
                "The converted SFT record has no environment tool calls, and the "
                "SDK agent did not reach a finished state for the comparable prompt."
            )
        elif not performed_tool_call and reached_finish:
            failure_reason = (
                "The SDK agent finished with a plain assistant message and did not "
                "perform any parsed tool call for this task."
            )
        elif reached_finish and performed_tool_call:
            failure_reason = (
                "The SDK agent reached a finished state but only performed finish/"
                "think calls or malformed tool calls, not a parsed environment tool "
                "call."
            )
        elif performed_tool_call:
            failure_reason = (
                "The SDK agent performed tool calls but did not reach a successful "
                "finished state within the replayed mock environment."
            )
        else:
            failure_reason = (
                "The SDK agent did not perform a parsed tool call before the run ended."
            )
    selected_user_text = "\n\n---\n\n".join(
        text_from_content(record["messages"][index].get("content"))
        for index in selected_message_indices
    )
    run_path.write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "record_id": record.get("id"),
                "model": MODEL,
                "selected_message_indices": selected_message_indices,
                "selected_user_index": selected_message_indices[-1],
                "selected_user_text": selected_user_text,
                "validation_mode": (
                    "OpenHands SDK Agent/Conversation with replayed mock tool "
                    "executors for non-finish/non-think tools"
                ),
                "mocked_tools": mocked_tools,
                "expected_tool_calls_in_record": expected_tool_calls,
                "expected_environment_tool_calls_in_record": (expected_environment_tool_calls),
                "expects_environment_tool_calls": expects_environment_tool_calls,
                "max_iterations": max_iterations,
                "final_status": str(final_status),
                "completed_as_expected": completed_as_expected,
                "completed_with_tool_execution": completed_with_tool_execution,
                "completed_with_any_tool_call": completed_with_any_tool_call,
                "performed_tool_call": performed_tool_call,
                "performed_environment_tool_call": performed_environment_tool_call,
                "reached_finish": reached_finish,
                "called_finish_tool": called_finish_tool,
                "validation_status": validation_status,
                "failure_reason": failure_reason,
                "action_names": action_names,
                "actions": actions,
                "observations": observations,
                "completion_written": completion_written,
                "error": error,
            },
            indent=2,
        )
        + "\n"
    )
    if not completed_as_expected:
        raise RuntimeError(
            f"{dataset_name} did not complete as expected; "
            f"status={final_status}, actions={action_names}"
        )


__all__ = ["run_dataset_validation"]
