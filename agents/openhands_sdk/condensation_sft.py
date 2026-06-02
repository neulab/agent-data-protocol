from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from collections.abc import Iterator, Sequence
from typing import Any

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
os.environ.setdefault("LOG_LEVEL", "ERROR")

from openhands.sdk import LLM, Agent, Conversation, LLMConvertibleEvent, Message, TextContent
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.context.condenser.utils import get_total_token_count
from openhands.sdk.context.view import View
from openhands.sdk.event import LLMConvertibleEvent as SDKEvent
from openhands.sdk.event import MessageEvent, SystemPromptEvent
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.llm.llm_response import LLMResponse
from openhands.sdk.tool import ToolDefinition
from pydantic import PrivateAttr, SecretStr, TypeAdapter, ValidationError

from agents.openhands_sdk.std_to_sft import (
    SDKEventBuilder,
    append_message_action,
    normalize_message_content,
    register_metadata_tools,
    sdk_tool_specs,
    serializable_tool,
)
from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.dataset_metadata import load_dataset_metadata
from schema.observation.image import ImageObservation
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.tool_call_links import backfill_adjacent_tool_call_links
from schema.trajectory import Trajectory

TRAJECTORY_CONTENT_ADAPTER = TypeAdapter(
    list[
        ApiAction | CodeAction | MessageAction | TextObservation | WebObservation | ImageObservation
    ]
)

DEFAULT_MAX_SIZE = 1_000_000


class PromptCapturingLLM(LLM):
    """LLM that records condenser prompts before delegating to LiteLLM."""

    _captured_messages: list[list[Message]] = PrivateAttr(default_factory=list)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._captured_messages = []

    @property
    def captured_messages(self) -> list[list[Message]]:
        return self._captured_messages

    def completion(
        self,
        messages: list[Message],
        tools: Sequence[ToolDefinition] | None = None,
        _return_metrics: bool = False,
        add_security_risk_prediction: bool = False,
        on_token: Any | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self._captured_messages.append(messages)
        return super().completion(
            messages=messages,
            tools=tools,
            _return_metrics=_return_metrics,
            add_security_risk_prediction=add_security_risk_prediction,
            on_token=on_token,
            **kwargs,
        )


def load_trajectory(line: str) -> Trajectory:
    data = json.loads(line)
    try:
        return Trajectory(**data)
    except ValidationError:
        content = data.get("content")
        if not isinstance(content, list):
            raise
        data["content"] = backfill_adjacent_tool_call_links(
            TRAJECTORY_CONTENT_ADAPTER.validate_python(content)
        )
        return Trajectory(**data)


def format_messages(llm: LLM, messages: list[Message]) -> list[dict[str, Any]]:
    return normalize_message_content(llm.format_messages_for_llm(messages))


class TrackingSDKEventBuilder(SDKEventBuilder):
    def __init__(
        self,
        conversation: Conversation,
        metadata: Any,
        event_history: list[SDKEvent],
    ) -> None:
        super().__init__(conversation, metadata)
        self.event_history = event_history

    def append(self, event: SDKEvent) -> None:
        self.event_history.append(event)
        super().append(event)


def token_count(view: View, llm: LLM) -> int:
    return get_total_token_count(view.events, llm)


def make_condensation_prompt_record(
    *,
    trajectory_id: str,
    dataset_name: str | None,
    prompt_messages: list[Message],
    formatting_llm: LLM,
    condensation: Condensation,
    condensation_index: int,
    max_tokens: int,
    prompt_token_count: int,
) -> dict[str, Any]:
    if condensation.summary is None:
        raise RuntimeError("Condenser LLM did not return a summary")
    messages = [
        *prompt_messages,
        Message(role="assistant", content=[TextContent(text=condensation.summary)]),
    ]
    return {
        "id": f"{trajectory_id}__condensation_{condensation_index:04d}",
        "messages": format_messages(formatting_llm, messages),
        "tools": [],
        "metadata": {
            "agent": "openhands_sdk",
            "format": "openai_chat_completions",
            "source_dataset": dataset_name,
            "generation": "openhands_sdk_condensation_prompt",
            "source_trajectory_id": trajectory_id,
            "condensation_index": condensation_index,
            "max_tokens": max_tokens,
            "prompt_token_count_before_condensation": prompt_token_count,
            "forgotten_event_count": len(condensation.forgotten_event_ids),
            "summary_offset": condensation.summary_offset,
            "condensation_output": "llm",
        },
    }


def make_trajectory_record_from_conversation(
    *,
    conversation: Conversation,
    trajectory_id: str,
    dataset_name: str | None,
    segment_index: int,
    events: Sequence[Any] | None = None,
) -> dict[str, Any]:
    view = View.from_events(events if events is not None else conversation.state.events)
    messages = LLMConvertibleEvent.events_to_messages(view.events)
    tools = [serializable_tool(tool) for tool in conversation.agent.tools_map.values()]
    return {
        "id": f"{trajectory_id}__trajectory_{segment_index:04d}",
        "messages": format_messages(conversation.agent.llm, messages),
        "tools": tools,
        "metadata": {
            "agent": "openhands_sdk",
            "format": "openai_chat_completions",
            "source_dataset": dataset_name,
            "generation": "openhands_sdk_events",
            "record_type": "trajectory",
            "source_trajectory_id": trajectory_id,
            "trajectory_segment_index": segment_index,
        },
    }


def condensation_prompt_record_if_needed(
    *,
    events: list[SDKEvent],
    condenser: LLMSummarizingCondenser,
    agent_llm: LLM,
    condenser_llm: PromptCapturingLLM,
    trajectory_id: str,
    dataset_name: str | None,
    max_tokens: int,
    condensation_index: int,
) -> tuple[Condensation, dict[str, Any]] | None:
    view = View.from_events(events)
    prompt_token_count = token_count(view, condenser.llm)
    before_prompt_count = len(condenser_llm.captured_messages)
    condensation_result = condenser.condense(view, agent_llm=agent_llm)
    if not isinstance(condensation_result, Condensation):
        return None

    if len(condenser_llm.captured_messages) != before_prompt_count + 1:
        raise RuntimeError("Condenser returned Condensation without calling its LLM")

    prompt_record = make_condensation_prompt_record(
        trajectory_id=trajectory_id,
        dataset_name=dataset_name,
        prompt_messages=condenser_llm.captured_messages[-1],
        formatting_llm=agent_llm,
        condensation=condensation_result,
        condensation_index=condensation_index,
        max_tokens=max_tokens,
        prompt_token_count=prompt_token_count,
    )
    return condensation_result, prompt_record


def append_standardized_events_with_condensation(
    *,
    conversation: Conversation,
    trajectory: Trajectory,
    dataset_name: str | None,
    max_tokens: int,
    model: str,
    max_size: int,
    keep_first: int,
    start_index: int,
    include_trajectories: bool,
) -> list[dict[str, Any]]:
    metadata = load_dataset_metadata(dataset_name, required=True)
    event_history: list[SDKEvent] = [
        SystemPromptEvent(
            system_prompt=TextContent(text=conversation.agent.static_system_message),
            tools=list(conversation.agent.tools_map.values()),
        )
    ]
    builder = TrackingSDKEventBuilder(conversation, metadata, event_history)
    first_event = trajectory.content[0]
    if not isinstance(first_event, TextObservation) or first_event.source != "user":
        raise ValueError(
            "OpenHands SDK condensation conversion expects the first event to be a "
            "user TextObservation"
        )
    builder.append(
        MessageEvent(
            source="user",
            llm_message=Message(
                role="user",
                content=[TextContent(text=first_event.content)],
            ),
        )
    )
    condenser_llm = PromptCapturingLLM(
        usage_id="openhands-sdk-condensation-sft-condenser",
        model=model,
        api_key=SecretStr(os.getenv("LLM_API_KEY") or "not-used"),
        base_url=os.getenv("LLM_BASE_URL"),
    )
    condenser = LLMSummarizingCondenser(
        llm=condenser_llm,
        max_size=max_size,
        max_tokens=max_tokens,
        keep_first=keep_first,
    )
    records: list[dict[str, Any]] = []
    segment_index = 1
    condensation_index = 1
    index = start_index
    batch_number = 0
    last_safe_events = list(event_history)

    def update_last_safe_events() -> None:
        nonlocal last_safe_events
        view = View.from_events(event_history)
        if token_count(view, conversation.agent.llm) <= max_tokens:
            last_safe_events = list(event_history)

    def emit_condensation_boundary_if_needed() -> None:
        nonlocal segment_index, condensation_index, last_safe_events
        result = condensation_prompt_record_if_needed(
            events=event_history,
            condenser=condenser,
            agent_llm=conversation.agent.llm,
            condenser_llm=condenser_llm,
            trajectory_id=trajectory.id,
            dataset_name=dataset_name,
            max_tokens=max_tokens,
            condensation_index=condensation_index,
        )
        if result is None:
            return
        condensation, prompt_record = result
        if include_trajectories:
            records.append(
                make_trajectory_record_from_conversation(
                    conversation=conversation,
                    trajectory_id=trajectory.id,
                    dataset_name=dataset_name,
                    segment_index=segment_index,
                    events=last_safe_events,
                )
            )
            segment_index += 1
        records.append(prompt_record)
        event_history.append(condensation)
        conversation.state.events.append(condensation)
        last_safe_events = list(event_history)
        condensation_index += 1

    while index < len(trajectory.content):
        event = trajectory.content[index]
        if isinstance(event, (ApiAction, CodeAction)):
            emit_condensation_boundary_if_needed()
            action_batch: list[ApiAction | CodeAction] = []
            while index < len(trajectory.content) and isinstance(
                trajectory.content[index], (ApiAction, CodeAction)
            ):
                action_batch.append(trajectory.content[index])
                index += 1
            batch_number += 1
            builder.append_action_batch(action_batch, batch_number=batch_number)
            update_last_safe_events()
            continue

        if isinstance(event, MessageAction):
            emit_condensation_boundary_if_needed()
            append_message_action(builder, event)
            update_last_safe_events()
        elif isinstance(event, (TextObservation, WebObservation, ImageObservation)):
            builder.append_observation(event)
            update_last_safe_events()
        else:
            raise ValueError(f"Unsupported event type: {type(event)}")
        index += 1

    emit_condensation_boundary_if_needed()
    if include_trajectories or not records:
        records.append(
            make_trajectory_record_from_conversation(
                conversation=conversation,
                trajectory_id=trajectory.id,
                dataset_name=dataset_name,
                segment_index=segment_index,
                events=last_safe_events,
            )
        )

    return records


def process_row(
    line: str,
    *,
    max_tokens: int,
    model: str,
    dataset_name: str | None = None,
    include_trajectories: bool = True,
    max_size: int = DEFAULT_MAX_SIZE,
    keep_first: int = 2,
) -> list[dict[str, Any]]:
    trajectory = load_trajectory(line)
    dataset_name = dataset_name or os.getenv("MY_DATASET")
    metadata = load_dataset_metadata(dataset_name, required=True)
    register_metadata_tools(metadata)
    first_event = trajectory.content[0] if trajectory.content else None
    if not isinstance(first_event, TextObservation) or first_event.source != "user":
        raise ValueError(
            "OpenHands SDK condensation conversion expects the first event to be a "
            "user TextObservation"
        )

    llm = LLM(
        usage_id="openhands-sdk-condensation-sft-converter",
        model=model,
        api_key=SecretStr(os.getenv("LLM_API_KEY") or "not-used"),
        base_url=os.getenv("LLM_BASE_URL"),
    )
    agent = Agent(llm=llm, tools=sdk_tool_specs(trajectory, metadata))
    with tempfile.TemporaryDirectory(prefix="openhands-sdk-condensation-sft-") as tmpdir:
        conversation = Conversation(agent=agent, workspace=tmpdir, visualizer=None)
        try:
            conversation._ensure_agent_ready()
            return append_standardized_events_with_condensation(
                conversation=conversation,
                trajectory=trajectory,
                dataset_name=dataset_name,
                max_tokens=max_tokens,
                model=model,
                max_size=max_size,
                keep_first=keep_first,
                start_index=1,
                include_trajectories=include_trajectories,
            )
        finally:
            conversation.close()


def iter_input_chunks(chunk_size: int) -> Iterator[list[str]]:
    chunk: list[str] = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        chunk.append(line)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


async def process_line(
    line: str,
    *,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    try:
        async with semaphore:
            return await asyncio.to_thread(
                process_row,
                line,
                max_tokens=args.max_tokens,
                model=args.model,
                include_trajectories=args.include_trajectories == "yes",
                max_size=args.max_size,
                keep_first=args.keep_first,
            )
    except Exception as exc:
        if not args.continue_on_error:
            raise
        row_id = None
        try:
            row_id = json.loads(line).get("id")
        except Exception:
            pass
        print(
            json.dumps(
                {
                    "id": row_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        return []


async def process_stream(args: argparse.Namespace) -> None:
    from tqdm import tqdm

    semaphore = asyncio.Semaphore(args.concurrency)
    progress = tqdm(
        desc="condensation_sft",
        unit="row",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    try:
        for chunk in iter_input_chunks(args.chunk_size):
            tasks = [
                asyncio.create_task(process_line(line, args=args, semaphore=semaphore))
                for line in chunk
            ]
            for task in asyncio.as_completed(tasks):
                records = await task
                for record in records:
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                progress.update(1)
    finally:
        progress.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Emit OpenHands SDK trajectory SFT records plus LLM-generated condenser "
            "summary records whenever replayed ADP trajectories exceed a token threshold."
        )
    )
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE)
    parser.add_argument("--keep-first", type=int, default=2)
    parser.add_argument(
        "--include-trajectories",
        choices=["yes", "no"],
        default="yes",
        help="Whether to emit the original OpenHands SDK trajectory record before summaries.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of input trajectories to process concurrently.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Number of input rows to schedule per async batch.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress output on stderr.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log per-row conversion errors to stderr and continue processing remaining rows.",
    )
    args = parser.parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")
    asyncio.run(process_stream(args))


if __name__ == "__main__":
    main()
