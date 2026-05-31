from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from typing import Any

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
os.environ.setdefault("LOG_LEVEL", "ERROR")

from litellm.types.utils import Choices, ModelResponse
from litellm.types.utils import Message as LiteLLMMessage
from openhands.sdk import LLM, Agent, Conversation, LLMConvertibleEvent, Message, TextContent
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.context.condenser.utils import get_total_token_count
from openhands.sdk.context.view import View
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.llm.llm_response import LLMResponse
from openhands.sdk.llm.utils.metrics import MetricsSnapshot, TokenUsage
from openhands.sdk.tool import ToolDefinition
from pydantic import PrivateAttr, SecretStr

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
from schema.trajectory import Trajectory

DEFAULT_CONDENSATION_SUMMARY_TEMPLATE = "[ADP condensation placeholder #{index}]"
DEFAULT_MAX_SIZE = 1_000_000
CONDENSATION_OUTPUT_MODES = {"none", "placeholder", "llm"}


class PromptCapturingLLM(LLM):
    """LLM that records condenser prompts and optionally delegates to LiteLLM."""

    _summary_template: str = PrivateAttr(default=DEFAULT_CONDENSATION_SUMMARY_TEMPLATE)
    _output_mode: str = PrivateAttr(default="none")
    _captured_messages: list[list[Message]] = PrivateAttr(default_factory=list)

    def __init__(self, *, summary_template: str, output_mode: str, **data: Any) -> None:
        super().__init__(**data)
        self._summary_template = summary_template
        self._output_mode = output_mode
        self._captured_messages = []

    @property
    def captured_messages(self) -> list[list[Message]]:
        return self._captured_messages

    def completion(
        self,
        messages: list[Message],
        tools: Sequence[ToolDefinition] | None = None,  # noqa: ARG002
        _return_metrics: bool = False,  # noqa: ARG002
        add_security_risk_prediction: bool = False,  # noqa: ARG002
        on_token: Any | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> LLMResponse:
        self._captured_messages.append(messages)
        if self._output_mode == "llm":
            return super().completion(
                messages=messages,
                tools=tools,
                _return_metrics=_return_metrics,
                add_security_risk_prediction=add_security_risk_prediction,
                on_token=on_token,
                **kwargs,
            )

        index = len(self._captured_messages)
        summary = self._summary_template.format(index=index)
        response_message = Message(
            role="assistant",
            content=[TextContent(text=summary)],
        )
        raw_message = LiteLLMMessage(role="assistant", content=summary)
        raw_response = ModelResponse(
            id=f"adp-condensation-placeholder-{index:06d}",
            choices=[Choices(message=raw_message, index=0, finish_reason="stop")],
            created=0,
            model=self.model,
            object="chat.completion",
        )
        return LLMResponse(
            message=response_message,
            metrics=MetricsSnapshot(
                model_name=self.model,
                accumulated_cost=0.0,
                max_budget_per_task=None,
                accumulated_token_usage=TokenUsage(
                    model=self.model,
                    prompt_tokens=0,
                    completion_tokens=0,
                ),
            ),
            raw_response=raw_response,
        )


def format_messages(llm: LLM, messages: list[Message]) -> list[dict[str, Any]]:
    return normalize_message_content(llm.format_messages_for_llm(messages))


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
    output_mode: str,
) -> dict[str, Any]:
    messages = list(prompt_messages)
    if output_mode != "none" and condensation.summary is not None:
        messages.append(Message(role="assistant", content=[TextContent(text=condensation.summary)]))
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
            "condensation_output": output_mode,
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
    conversation: Conversation,
    condenser: LLMSummarizingCondenser,
    agent_llm: LLM,
    condenser_llm: PromptCapturingLLM,
    trajectory_id: str,
    dataset_name: str | None,
    max_tokens: int,
    condensation_index: int,
    condensation_output: str,
) -> tuple[Condensation, dict[str, Any]] | None:
    view = View.from_events(conversation.state.events)
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
        output_mode=condensation_output,
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
    summary_template: str,
    condensation_output: str,
    start_index: int,
    include_trajectories: bool,
) -> list[dict[str, Any]]:
    metadata = load_dataset_metadata(dataset_name, required=True)
    builder = SDKEventBuilder(conversation, metadata)
    condenser_llm = PromptCapturingLLM(
        usage_id="openhands-sdk-condensation-sft-condenser",
        model=model,
        api_key=SecretStr(os.getenv("LLM_API_KEY") or "not-used"),
        base_url=os.getenv("LLM_BASE_URL"),
        summary_template=summary_template,
        output_mode=condensation_output,
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
    last_safe_events = list(conversation.state.events)

    def update_last_safe_events() -> None:
        nonlocal last_safe_events
        view = View.from_events(conversation.state.events)
        if token_count(view, conversation.agent.llm) <= max_tokens:
            last_safe_events = list(conversation.state.events)

    def emit_condensation_boundary_if_needed() -> None:
        nonlocal segment_index, condensation_index, last_safe_events
        result = condensation_prompt_record_if_needed(
            conversation=conversation,
            condenser=condenser,
            agent_llm=conversation.agent.llm,
            condenser_llm=condenser_llm,
            trajectory_id=trajectory.id,
            dataset_name=dataset_name,
            max_tokens=max_tokens,
            condensation_index=condensation_index,
            condensation_output=condensation_output,
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
        conversation.state.events.append(condensation)
        last_safe_events = list(conversation.state.events)
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
    summary_template: str = DEFAULT_CONDENSATION_SUMMARY_TEMPLATE,
    condensation_output: str = "none",
) -> list[dict[str, Any]]:
    if condensation_output not in CONDENSATION_OUTPUT_MODES:
        raise ValueError(f"condensation_output must be one of {sorted(CONDENSATION_OUTPUT_MODES)}")
    trajectory = Trajectory(**json.loads(line))
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
            conversation.send_message(first_event.content)
            return append_standardized_events_with_condensation(
                conversation=conversation,
                trajectory=trajectory,
                dataset_name=dataset_name,
                max_tokens=max_tokens,
                model=model,
                max_size=max_size,
                keep_first=keep_first,
                summary_template=summary_template,
                condensation_output=condensation_output,
                start_index=1,
                include_trajectories=include_trajectories,
            )
        finally:
            conversation.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Emit OpenHands SDK trajectory SFT records plus condenser prompt records "
            "whenever replayed ADP trajectories exceed a token threshold."
        )
    )
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE)
    parser.add_argument("--keep-first", type=int, default=2)
    parser.add_argument("--summary-template", default=DEFAULT_CONDENSATION_SUMMARY_TEMPLATE)
    parser.add_argument(
        "--condensation-output",
        choices=sorted(CONDENSATION_OUTPUT_MODES),
        default="none",
        help=(
            "Whether condensation prompt records contain no assistant output, a "
            "deterministic placeholder output, or a real LLM output generated via "
            "LLM_MODEL/LLM_API_KEY/LLM_BASE_URL."
        ),
    )
    parser.add_argument(
        "--include-trajectories",
        choices=["yes", "no"],
        default="yes",
        help="Whether to emit the original OpenHands SDK trajectory record before prompts.",
    )
    args = parser.parse_args()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        records = process_row(
            line,
            max_tokens=args.max_tokens,
            model=args.model,
            include_trajectories=args.include_trajectories == "yes",
            max_size=args.max_size,
            keep_first=args.keep_first,
            summary_template=args.summary_template,
            condensation_output=args.condensation_output,
        )
        for record in records:
            print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
