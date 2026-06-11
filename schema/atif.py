from __future__ import annotations

import json
import re
from typing import Any, Literal, Union, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ATIF_SCHEMA_VERSION = "ATIF-v1.7"


class ImageSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    media_type: str | None = None
    path: str


class ContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text", "image"]
    text: str | None = None
    source: ImageSource | None = None
    extra: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_content_shape(self):
        if self.type == "text" and self.text is None:
            raise ValueError("text content parts require text")
        if self.type == "image" and self.source is None:
            raise ValueError("image content parts require source")
        return self


ATIFContent = Union[str, list[ContentPart]]


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    function_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] | None = None


class ObservationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_call_id: str | None = None
    content: ATIFContent
    subagent_trajectory_ref: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None


class ATIFObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[ObservationResult] = Field(default_factory=list)
    extra: dict[str, Any] | None = None


class Metrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost: float | None = None
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[Any] | None = None
    extra: dict[str, Any] | None = None


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: int
    timestamp: str | None = None
    source: Literal["system", "user", "agent"]
    message: ATIFContent = ""
    model_name: str | None = None
    reasoning_effort: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    observation: ATIFObservation | None = None
    metrics: Metrics | None = None
    llm_call_count: int | None = None
    is_copied_context: bool | None = None
    extra: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_agent_fields_and_links(self):
        if self.source != "agent":
            if self.tool_calls:
                raise ValueError("tool_calls are only valid on agent steps")
            if self.reasoning_content is not None or self.reasoning_effort is not None:
                raise ValueError("reasoning fields are only valid on agent steps")

        tool_call_ids = {tool_call.tool_call_id for tool_call in self.tool_calls or []}
        for result in self.observation.results if self.observation else []:
            if result.source_call_id is not None and result.source_call_id not in tool_call_ids:
                raise ValueError(
                    f"observation result source_call_id {result.source_call_id!r} does not "
                    "match a tool call in the same step"
                )
        return self


class Agent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "atif"
    version: str = "unknown"
    model_name: str | None = None
    tool_definitions: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None


class ATIFTrajectory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = ATIF_SCHEMA_VERSION
    session_id: str | None = None
    trajectory_id: str | None = None
    agent: Agent = Field(default_factory=Agent)
    steps: list[Step] = Field(..., min_length=1)
    notes: str | None = None
    final_metrics: dict[str, Any] | None = None
    continued_trajectory_ref: str | None = None
    extra: dict[str, Any] | None = None
    subagent_trajectories: list[dict[str, Any]] | None = None

    @field_validator("schema_version")
    def validate_schema_version(cls, value: str) -> str:
        if value != ATIF_SCHEMA_VERSION:
            raise ValueError(f"Unsupported ATIF schema_version {value!r}")
        return value

    @model_validator(mode="after")
    def validate_sequential_step_ids(self):
        step_ids = [step.step_id for step in self.steps]
        expected = list(range(1, len(self.steps) + 1))
        if step_ids != expected:
            raise ValueError(f"ATIF step_id values must be sequential starting at 1: {step_ids}")
        return self


def content_to_text(content: ATIFContent) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if part.type == "text" and part.text is not None:
            parts.append(part.text)
        elif part.type == "image" and part.source is not None:
            parts.append(f"[Image: {part.source.path}]")
    return "\n".join(parts)


def normalize_atif_trajectory(trajectory: ATIFTrajectory) -> ATIFTrajectory:
    normalized = cast(ATIFTrajectory, trajectory.model_copy(deep=True))
    seen_tool_call_ids: set[str] = set()
    next_tool_call_ordinal = 1

    def unique_tool_call_id(tool_call_id: str) -> str:
        nonlocal next_tool_call_ordinal
        if tool_call_id not in seen_tool_call_ids:
            seen_tool_call_ids.add(tool_call_id)
            return tool_call_id
        while True:
            candidate = f"call_{next_tool_call_ordinal:06d}"
            next_tool_call_ordinal += 1
            if candidate not in seen_tool_call_ids:
                seen_tool_call_ids.add(candidate)
                return candidate

    for step in normalized.steps:
        if step.source == "agent" and isinstance(step.message, str):
            think_match = re.search(r"<think>(.*?)</think>", step.message, flags=re.DOTALL)
            if think_match:
                reasoning = think_match.group(1).strip()
                remaining = (
                    step.message[: think_match.start()] + step.message[think_match.end() :]
                ).strip()
                step.message = remaining
                if reasoning and not step.reasoning_content:
                    step.reasoning_content = reasoning

        rewritten_call_ids: dict[str, str] = {}
        for tool_call in step.tool_calls or []:
            original_tool_call_id = tool_call.tool_call_id
            tool_call.tool_call_id = unique_tool_call_id(tool_call.tool_call_id)
            if tool_call.tool_call_id != original_tool_call_id:
                rewritten_call_ids[original_tool_call_id] = tool_call.tool_call_id

            function_name = tool_call.function_name
            arguments = dict(tool_call.arguments)
            lower_name = function_name.lower()
            normalized_language = None
            if lower_name in {"bash", "shell", "sh", "terminal"}:
                normalized_language = "bash"
                tool_call.function_name = "execute_bash"
                if "command" not in arguments:
                    arguments = {"command": arguments.get("code") or arguments.get("content") or ""}
            elif lower_name in {
                "add_and_execute_jupyter_code_cell",
                "execute",
                "python",
                "py",
                "python3",
                "ipython",
            }:
                normalized_language = "python"
                tool_call.function_name = "execute_ipython_cell"
                if "code" not in arguments:
                    arguments = {"code": arguments.get("command") or arguments.get("content") or ""}
                else:
                    arguments = {"code": arguments.get("code") or ""}
            elif lower_name == "execute_code":
                language = str(arguments.get("language", "")).lower()
                if language in {"bash", "sh", "shell"}:
                    normalized_language = "bash"
                    tool_call.function_name = "execute_bash"
                    arguments = {"command": arguments.get("content") or arguments.get("code") or ""}
                elif language in {"python", "py", "python3"}:
                    normalized_language = "python"
                    tool_call.function_name = "execute_ipython_cell"
                    arguments = {"code": arguments.get("content") or arguments.get("command") or ""}
            elif lower_name == "final_answer":
                tool_call.function_name = "finish"
                arguments = {
                    "message": arguments.get("answer") or arguments.get("message") or "",
                    "task_completed": "true",
                }
            elif lower_name == "localization_finish":
                tool_call.function_name = "finish"
                arguments = {
                    "message": json.dumps(arguments.get("locations") or arguments),
                    "task_completed": "true",
                }
            if normalized_language and tool_call.extra and "language" in tool_call.extra:
                tool_call.extra = {**tool_call.extra, "language": normalized_language}
            tool_call.arguments = arguments
        if step.observation:
            for result in step.observation.results:
                if result.source_call_id in rewritten_call_ids:
                    result.source_call_id = rewritten_call_ids[result.source_call_id]
    return normalized
