from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Sequence
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
from pydantic import PrivateAttr, SecretStr
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from agents.openhands_sdk.std_to_sft import (
    SDKEventBuilder,
    append_message_action,
    normalize_message_content,
    register_metadata_tools,
    sdk_tool_specs,
    serializable_tool,
)
from schema.dataset_metadata import load_dataset_metadata
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

DEFAULT_MAX_SIZE = 1_000_000


def source_row_id_from_line(line: str, trajectory_id: str) -> str:
    row = json.loads(line)
    canonical = json.dumps(
        row,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{trajectory_id}__row_{digest}"


class PromptCapturingLLM(LLM):
    """LLM that records condenser prompts before delegating to LiteLLM."""

    _captured_messages: list[list[Message]] = PrivateAttr(default_factory=list)
    _completion_semaphore: asyncio.Semaphore | None = PrivateAttr(default=None)
    _llm_retries: int = PrivateAttr(default=1)
    _llm_retry_min_wait: float = PrivateAttr(default=1.0)
    _llm_retry_max_wait: float = PrivateAttr(default=30.0)

    def __init__(self, **data: Any) -> None:
        completion_semaphore = data.pop("completion_semaphore", None)
        llm_retries = data.pop(
            "llm_retries",
            int(os.getenv("ADP_CONDENSER_LLM_RETRIES", "3")),
        )
        llm_retry_min_wait = data.pop(
            "llm_retry_min_wait",
            float(os.getenv("ADP_CONDENSER_LLM_RETRY_MIN_WAIT", "1")),
        )
        llm_retry_max_wait = data.pop(
            "llm_retry_max_wait",
            float(os.getenv("ADP_CONDENSER_LLM_RETRY_MAX_WAIT", "30")),
        )
        super().__init__(**data)
        self._captured_messages = []
        self._completion_semaphore = completion_semaphore
        self._llm_retries = max(1, llm_retries)
        self._llm_retry_min_wait = max(0, llm_retry_min_wait)
        self._llm_retry_max_wait = max(self._llm_retry_min_wait, llm_retry_max_wait)

    @property
    def captured_messages(self) -> list[list[Message]]:
        return self._captured_messages

    async def acompletion(
        self,
        messages: list[Message],
        tools: Sequence[ToolDefinition] | None = None,
        _return_metrics: bool = False,
        add_security_risk_prediction: bool = False,
        on_token: Any | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self._captured_messages.append(messages)

        async def run_completion() -> LLMResponse:
            return await super(PromptCapturingLLM, self).acompletion(
                messages=messages,
                tools=tools,
                _return_metrics=_return_metrics,
                add_security_risk_prediction=add_security_risk_prediction,
                on_token=on_token,
                **kwargs,
            )

        async def run_with_retries() -> LLMResponse:
            if self._llm_retries <= 1:
                return await run_completion()
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._llm_retries),
                wait=wait_exponential_jitter(
                    initial=self._llm_retry_min_wait,
                    max=self._llm_retry_max_wait,
                ),
                retry=retry_if_exception_type(Exception),
                reraise=True,
            ):
                with attempt:
                    return await run_completion()
            raise RuntimeError("unreachable retry state")

        if self._completion_semaphore is None:
            return await run_with_retries()
        async with self._completion_semaphore:
            return await run_with_retries()


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


def formatted_token_count(events: Sequence[SDKEvent], llm: LLM) -> int:
    view = View.from_events(events)
    messages = LLMConvertibleEvent.events_to_messages(view.events)
    return llm.get_token_count(messages)


def make_condensation_prompt_record(
    *,
    trajectory_id: str,
    source_trajectory_id: str | None = None,
    source_row_id: str | None = None,
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
    metadata = {
        "agent": "openhands_sdk",
        "format": "openai_chat_completions",
        "source_dataset": dataset_name,
        "generation": "openhands_sdk_condensation_prompt",
        "source_trajectory_id": source_trajectory_id or trajectory_id,
        "condensation_index": condensation_index,
        "max_tokens": max_tokens,
        "prompt_token_count_before_condensation": prompt_token_count,
        "forgotten_event_count": len(condensation.forgotten_event_ids),
        "summary_offset": condensation.summary_offset,
        "condensation_output": "llm",
    }
    if source_row_id is not None:
        metadata["source_row_id"] = source_row_id
    return {
        "id": f"{trajectory_id}__condensation_{condensation_index:04d}",
        "messages": format_messages(formatting_llm, messages),
        "tools": [],
        "metadata": metadata,
    }


def make_trajectory_record_from_conversation(
    *,
    conversation: Conversation,
    trajectory_id: str,
    source_trajectory_id: str | None = None,
    source_row_id: str | None = None,
    dataset_name: str | None,
    segment_index: int,
    events: Sequence[Any] | None = None,
) -> dict[str, Any]:
    view = View.from_events(events if events is not None else conversation.state.events)
    messages = LLMConvertibleEvent.events_to_messages(view.events)
    tools = [serializable_tool(tool) for tool in conversation.agent.tools_map.values()]
    metadata = {
        "agent": "openhands_sdk",
        "format": "openai_chat_completions",
        "source_dataset": dataset_name,
        "generation": "openhands_sdk_events",
        "record_type": "trajectory",
        "source_trajectory_id": source_trajectory_id or trajectory_id,
        "trajectory_segment_index": segment_index,
    }
    if source_row_id is not None:
        metadata["source_row_id"] = source_row_id
    return {
        "id": f"{trajectory_id}__trajectory_{segment_index:04d}",
        "messages": format_messages(conversation.agent.llm, messages),
        "tools": tools,
        "metadata": metadata,
    }


async def acondensation_prompt_record_if_needed(
    *,
    events: list[SDKEvent],
    condenser: LLMSummarizingCondenser,
    agent_llm: LLM,
    condenser_llm: PromptCapturingLLM,
    trajectory_id: str,
    source_trajectory_id: str | None = None,
    source_row_id: str | None = None,
    dataset_name: str | None,
    max_tokens: int,
    condensation_index: int,
) -> tuple[Condensation, dict[str, Any]] | None:
    view = View.from_events(events)
    prompt_token_count = token_count(view, condenser.llm)
    before_prompt_count = len(condenser_llm.captured_messages)
    condensation_result = await condenser.acondense(view, agent_llm=agent_llm)
    if not isinstance(condensation_result, Condensation):
        return None

    if len(condenser_llm.captured_messages) != before_prompt_count + 1:
        raise RuntimeError("Condenser returned Condensation without calling its LLM")

    prompt_record = make_condensation_prompt_record(
        trajectory_id=trajectory_id,
        source_trajectory_id=source_trajectory_id,
        source_row_id=source_row_id,
        dataset_name=dataset_name,
        prompt_messages=condenser_llm.captured_messages[-1],
        formatting_llm=agent_llm,
        condensation=condensation_result,
        condensation_index=condensation_index,
        max_tokens=max_tokens,
        prompt_token_count=prompt_token_count,
    )
    return condensation_result, prompt_record


async def append_standardized_events_with_condensation_async(
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
    output_trajectory_id: str | None = None,
    source_row_id: str | None = None,
    llm_semaphore: asyncio.Semaphore | None = None,
) -> list[dict[str, Any]]:
    """Replay trajectory events and emit trajectory plus condensation records.

    ``start_index`` is respected only when the first trajectory event is a user
    ``TextObservation`` that has already been emitted as the opening user
    message. On the fallback path, where there is no leading user message to
    consume, the index is intentionally reset to 0 so no trajectory content is
    skipped.
    """
    metadata = load_dataset_metadata(dataset_name, required=True)
    event_history: list[SDKEvent] = [
        SystemPromptEvent(
            system_prompt=TextContent(text=conversation.agent.static_system_message),
            tools=list(conversation.agent.tools_map.values()),
        )
    ]
    builder = TrackingSDKEventBuilder(conversation, metadata, event_history)
    first_event = trajectory.content[0]
    index = start_index
    if isinstance(first_event, TextObservation) and first_event.source == "user":
        builder.append(
            MessageEvent(
                source="user",
                llm_message=Message(
                    role="user",
                    content=[TextContent(text=first_event.content)],
                ),
            )
        )
    else:
        builder.append(
            MessageEvent(
                source="user",
                llm_message=Message(
                    role="user",
                    content=[
                        TextContent(text="Continue the task from the current workspace state.")
                    ],
                ),
            )
        )
        index = 0
    condenser_llm = PromptCapturingLLM(
        usage_id="openhands-sdk-condensation-sft-condenser",
        model=model,
        api_key=SecretStr(os.getenv("LLM_API_KEY") or "not-used"),
        base_url=os.getenv("LLM_BASE_URL"),
        completion_semaphore=llm_semaphore,
    )
    condenser = LLMSummarizingCondenser(
        llm=condenser_llm,
        max_size=max_size,
        max_tokens=max_tokens,
        keep_first=keep_first,
    )
    records: list[dict[str, Any]] = []
    record_trajectory_id = output_trajectory_id or trajectory.id
    segment_index = 1
    condensation_index = 1
    batch_number = 0
    last_safe_events = list(event_history)

    def update_last_safe_events() -> None:
        nonlocal last_safe_events
        if formatted_token_count(event_history, conversation.agent.llm) <= max_tokens:
            last_safe_events = list(event_history)

    async def emit_condensation_boundary_if_needed() -> None:
        nonlocal segment_index, condensation_index, last_safe_events
        result = await acondensation_prompt_record_if_needed(
            events=event_history,
            condenser=condenser,
            agent_llm=conversation.agent.llm,
            condenser_llm=condenser_llm,
            trajectory_id=record_trajectory_id,
            source_trajectory_id=trajectory.id,
            source_row_id=source_row_id,
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
                    trajectory_id=record_trajectory_id,
                    source_trajectory_id=trajectory.id,
                    source_row_id=source_row_id,
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
            await emit_condensation_boundary_if_needed()
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
            await emit_condensation_boundary_if_needed()
            append_message_action(builder, event)
            update_last_safe_events()
        elif isinstance(event, (TextObservation, WebObservation, ImageObservation)):
            builder.append_observation(event)
            update_last_safe_events()
        else:
            raise ValueError(f"Unsupported event type: {type(event)}")
        index += 1

    await emit_condensation_boundary_if_needed()
    if include_trajectories or not records:
        records.append(
            make_trajectory_record_from_conversation(
                conversation=conversation,
                trajectory_id=record_trajectory_id,
                source_trajectory_id=trajectory.id,
                source_row_id=source_row_id,
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
    return asyncio.run(
        process_row_async(
            line,
            max_tokens=max_tokens,
            model=model,
            dataset_name=dataset_name,
            include_trajectories=include_trajectories,
            max_size=max_size,
            keep_first=keep_first,
        )
    )


async def process_row_async(
    line: str,
    *,
    max_tokens: int,
    model: str,
    dataset_name: str | None = None,
    include_trajectories: bool = True,
    max_size: int = DEFAULT_MAX_SIZE,
    keep_first: int = 2,
    llm_semaphore: asyncio.Semaphore | None = None,
) -> list[dict[str, Any]]:
    trajectory = load_trajectory(line)
    output_trajectory_id = trajectory.id
    source_row_id = None
    if os.getenv("ADP_USE_SOURCE_ROW_HASH") == "1":
        source_row_id = source_row_id_from_line(line, trajectory.id)
        output_trajectory_id = source_row_id
    dataset_name = dataset_name or os.getenv("MY_DATASET")
    metadata = load_dataset_metadata(dataset_name, required=True)
    register_metadata_tools(metadata)

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
            return await append_standardized_events_with_condensation_async(
                conversation=conversation,
                trajectory=trajectory,
                dataset_name=dataset_name,
                max_tokens=max_tokens,
                model=model,
                max_size=max_size,
                keep_first=keep_first,
                start_index=1,
                include_trajectories=include_trajectories,
                output_trajectory_id=output_trajectory_id,
                source_row_id=source_row_id,
                llm_semaphore=llm_semaphore,
            )
        finally:
            conversation.close()


async def process_line(
    line: str,
    *,
    args: argparse.Namespace,
    llm_semaphore: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    def row_id() -> Any:
        try:
            row = json.loads(line)
        except Exception:
            return None
        return row.get("id") or row.get("trajectory_id") or row.get("session_id")

    def log_error(error_type: str, error: str) -> None:
        print(
            json.dumps(
                {
                    "id": row_id(),
                    "error_type": error_type,
                    "error": error,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )

    row_task = asyncio.create_task(
        process_row_async(
            line,
            max_tokens=args.max_tokens,
            model=args.model,
            include_trajectories=args.include_trajectories == "yes",
            max_size=args.max_size,
            keep_first=args.keep_first,
            llm_semaphore=llm_semaphore,
        )
    )
    try:
        if args.row_timeout > 0:
            return await asyncio.wait_for(row_task, timeout=args.row_timeout)
        return await row_task
    except asyncio.CancelledError:
        row_task.cancel()
        await asyncio.gather(row_task, return_exceptions=True)
        raise
    except asyncio.TimeoutError:
        row_task.cancel()
        await asyncio.gather(row_task, return_exceptions=True)
        log_error("TimeoutError", f"row exceeded timeout={args.row_timeout}s")
        if args.continue_on_error:
            return []
        raise
    except Exception as exc:
        if not args.continue_on_error:
            raise
        log_error(type(exc).__name__, str(exc))
        return []


async def process_stream(args: argparse.Namespace) -> None:
    from tqdm import tqdm

    llm_semaphore = asyncio.Semaphore(args.llm_concurrency)
    progress = tqdm(
        desc="condensation_sft",
        unit="row",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    pending: set[asyncio.Task[list[dict[str, Any]]]] = set()
    input_exhausted = False
    input_iter = iter(sys.stdin)

    def schedule_available() -> None:
        nonlocal input_exhausted
        while len(pending) < args.max_in_flight_rows and not input_exhausted:
            for line in input_iter:
                line = line.strip()
                if line:
                    pending.add(
                        asyncio.create_task(
                            process_line(
                                line,
                                args=args,
                                llm_semaphore=llm_semaphore,
                            )
                        )
                    )
                    break
            else:
                input_exhausted = True

    try:
        schedule_available()
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    records = await task
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    print(
                        json.dumps(
                            {
                                "id": None,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            },
                            ensure_ascii=False,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    records = []
                for record in records:
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                progress.update(1)
            schedule_available()
    except Exception:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise
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
        "--chunk-size",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-in-flight-rows",
        type=int,
        default=int(os.getenv("ADP_CONDENSER_MAX_IN_FLIGHT_ROWS", "100")),
        help="Maximum number of input rows to keep scheduled at once.",
    )
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=int(os.getenv("ADP_CONDENSER_LLM_CONCURRENCY", "0")),
        help=(
            "Maximum concurrent async condenser LLM requests. Defaults to "
            "--max-in-flight-rows when unset or 0."
        ),
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
    parser.add_argument(
        "--row-timeout",
        type=float,
        default=float(os.getenv("ADP_CONDENSER_ROW_TIMEOUT", "1800")),
        help=(
            "Maximum seconds to spend on one input row before failing the run. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--llm-retries",
        type=int,
        default=int(os.getenv("ADP_CONDENSER_LLM_RETRIES", "3")),
        help="Maximum attempts for each async condenser LLM request.",
    )
    parser.add_argument(
        "--llm-retry-min-wait",
        type=float,
        default=float(os.getenv("ADP_CONDENSER_LLM_RETRY_MIN_WAIT", "1")),
        help="Minimum retry wait in seconds for async condenser LLM requests.",
    )
    parser.add_argument(
        "--llm-retry-max-wait",
        type=float,
        default=float(os.getenv("ADP_CONDENSER_LLM_RETRY_MAX_WAIT", "30")),
        help="Maximum retry wait in seconds for async condenser LLM requests.",
    )
    args = parser.parse_args()
    if args.chunk_size is not None:
        args.max_in_flight_rows = args.chunk_size
    if args.max_in_flight_rows < 1:
        raise ValueError("--max-in-flight-rows must be at least 1")
    if args.llm_concurrency < 0:
        raise ValueError("--llm-concurrency must be non-negative")
    if args.llm_concurrency == 0:
        args.llm_concurrency = args.max_in_flight_rows
    if args.row_timeout < 0:
        raise ValueError("--row-timeout must be non-negative")
    if args.llm_retries < 1:
        raise ValueError("--llm-retries must be at least 1")
    if args.llm_retry_min_wait < 0:
        raise ValueError("--llm-retry-min-wait must be non-negative")
    if args.llm_retry_max_wait < args.llm_retry_min_wait:
        raise ValueError("--llm-retry-max-wait must be at least --llm-retry-min-wait")
    os.environ["ADP_CONDENSER_LLM_RETRIES"] = str(args.llm_retries)
    os.environ["ADP_CONDENSER_LLM_RETRY_MIN_WAIT"] = str(args.llm_retry_min_wait)
    os.environ["ADP_CONDENSER_LLM_RETRY_MAX_WAIT"] = str(args.llm_retry_max_wait)
    try:
        asyncio.run(process_stream(args))
    except asyncio.TimeoutError:
        sys.exit(124)


if __name__ == "__main__":
    main()
