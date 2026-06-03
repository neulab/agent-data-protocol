from __future__ import annotations

from typing import Any, Literal, Union, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.image import ImageObservation
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.tool_call_links import create_trajectory_with_tool_call_links
from schema.trajectory import Trajectory as ADPTrajectory

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

    name: str = "adp"
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


def observation_to_atif_content(observation: Observation) -> ATIFContent:
    if isinstance(observation, TextObservation):
        return observation.content
    if isinstance(observation, ImageObservation):
        return [
            ContentPart(
                type="image",
                source=ImageSource(path=observation.content),
                extra={"annotations": _dump_or_none(observation.annotations)},
            )
        ]
    if isinstance(observation, WebObservation):
        return ""
    return ""


def _dump_or_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return [_dump_or_none(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return value


def _result_extra_from_observation(observation: Observation) -> dict[str, Any] | None:
    extra: dict[str, Any] = {
        "adp_class": getattr(observation, "class_", None),
        "adp_source": getattr(observation, "source", None),
    }
    reward = getattr(observation, "reward", None)
    if reward is not None:
        extra["reward"] = reward
    if isinstance(observation, ImageObservation):
        extra["annotations"] = _dump_or_none(observation.annotations)
    if isinstance(observation, WebObservation):
        extra["web"] = {
            "html": observation.html,
            "axtree": observation.axtree,
            "url": observation.url,
            "image_observation": _dump_or_none(observation.image_observation),
            "viewport_size": observation.viewport_size,
        }
    return {key: value for key, value in extra.items() if value is not None} or None


def _message_extra_from_action(
    action: ApiAction | CodeAction | MessageAction,
) -> dict[str, Any] | None:
    extra: dict[str, Any] = {"adp_class": action.class_}
    description = getattr(action, "description", None)
    reward = getattr(action, "reward", None)
    if description is not None:
        extra["description"] = description
    if reward is not None:
        extra["reward"] = reward
    return extra or None


def code_action_to_tool_call(action: CodeAction, tool_call_id: str) -> ToolCall:
    language = action.language
    if language in {"bash", "sh", "shell"}:
        function_name = "execute_bash"
        arguments = {"command": action.content}
    elif language in {"python", "py", "python3"}:
        function_name = "execute_ipython_cell"
        arguments = {"code": action.content}
    else:
        function_name = "execute_code"
        arguments = {"language": language, "content": action.content}
    return ToolCall(
        tool_call_id=tool_call_id,
        function_name=function_name,
        arguments=arguments,
        extra={"adp_class": action.class_, "language": language},
    )


def _find_matching_observation(
    content: list[Any], start_index: int, tool_call_id: str | None
) -> tuple[int | None, Observation | None]:
    if tool_call_id is None:
        return None, None
    for index in range(start_index + 1, len(content)):
        item = content[index]
        if isinstance(item, Observation) and item.tool_call_id == tool_call_id:
            return index, item
    return None, None


def adp_trajectory_to_atif(trajectory: ADPTrajectory) -> ATIFTrajectory:
    steps: list[Step] = []
    consumed_observation_indices: set[int] = set()

    for index, item in enumerate(trajectory.content):
        if index in consumed_observation_indices:
            continue
        step_id = len(steps) + 1

        if isinstance(item, TextObservation):
            if item.tool_call_id is not None:
                continue
            source = item.source if item.source in {"user", "agent"} else "agent"
            observation = None
            message = item.content if item.source in {"user", "agent"} else ""
            llm_call_count = None
            extra = {"adp_class": item.class_, "adp_source": item.source}
            if item.source == "environment":
                observation = ATIFObservation(
                    results=[
                        ObservationResult(
                            content=item.content,
                            extra=_result_extra_from_observation(item),
                        )
                    ]
                )
                llm_call_count = 0
            steps.append(
                Step(
                    step_id=step_id,
                    source=source,
                    message=message,
                    observation=observation,
                    llm_call_count=llm_call_count,
                    extra=extra,
                )
            )
            continue

        if isinstance(item, (ImageObservation, WebObservation)):
            if item.tool_call_id is not None:
                continue
            steps.append(
                Step(
                    step_id=step_id,
                    source="agent",
                    message="",
                    observation=ATIFObservation(
                        results=[
                            ObservationResult(
                                content=observation_to_atif_content(item),
                                extra=_result_extra_from_observation(item),
                            )
                        ]
                    ),
                    llm_call_count=0,
                    extra={"adp_class": item.class_, "adp_source": getattr(item, "source", None)},
                )
            )
            continue

        if isinstance(item, MessageAction):
            steps.append(
                Step(
                    step_id=step_id,
                    source="agent",
                    message=item.content,
                    reasoning_content=item.reasoning_content,
                    extra=_message_extra_from_action(item),
                )
            )
            continue

        if isinstance(item, (ApiAction, CodeAction)):
            tool_call_id = item.tool_call_id or f"call_{step_id:06d}"
            matching_observation_index, matching_observation = _find_matching_observation(
                trajectory.content, index, item.tool_call_id
            )
            if matching_observation_index is not None:
                consumed_observation_indices.add(matching_observation_index)
            tool_call = (
                ToolCall(
                    tool_call_id=tool_call_id,
                    function_name=item.function,
                    arguments=item.kwargs,
                    extra={"adp_class": item.class_},
                )
                if isinstance(item, ApiAction)
                else code_action_to_tool_call(item, tool_call_id)
            )
            result = None
            if matching_observation is not None:
                result = ObservationResult(
                    source_call_id=tool_call_id,
                    content=observation_to_atif_content(matching_observation),
                    extra=_result_extra_from_observation(matching_observation),
                )
            steps.append(
                Step(
                    step_id=step_id,
                    source="agent",
                    message=getattr(item, "description", None) or "",
                    reasoning_content=item.reasoning_content,
                    tool_calls=[tool_call],
                    observation=ATIFObservation(results=[result]) if result else None,
                    extra=_message_extra_from_action(item),
                )
            )
            continue

        raise ValueError(f"Unsupported ADP content item: {item}")

    return ATIFTrajectory(
        schema_version=ATIF_SCHEMA_VERSION,
        session_id=trajectory.id,
        trajectory_id=trajectory.id,
        agent=Agent(name="adp", version=trajectory.schema_version),
        steps=steps,
        extra={
            "adp_schema_version": trajectory.schema_version,
            "adp_available_apis": trajectory.available_apis,
            "adp_details": trajectory.details,
        },
    )


def _observation_from_atif_result(result: ObservationResult) -> Observation:
    extra = result.extra or {}
    web = extra.get("web")
    if web:
        image = web.get("image_observation")
        image_observation = ImageObservation(**image) if image else None
        return WebObservation(
            tool_call_id=result.source_call_id,
            html=web.get("html"),
            axtree=web.get("axtree"),
            url=web.get("url"),
            image_observation=image_observation,
            viewport_size=tuple(web["viewport_size"]) if web.get("viewport_size") else None,
        )

    if isinstance(result.content, list) and result.content:
        image_part = next((part for part in result.content if part.type == "image"), None)
        if image_part and image_part.source:
            return ImageObservation(
                tool_call_id=result.source_call_id,
                content=image_part.source.path,
                annotations=None,
                source=extra.get("adp_source") or "environment",
                reward=extra.get("reward"),
            )

    return TextObservation(
        tool_call_id=result.source_call_id,
        content=content_to_text(result.content),
        source=extra.get("adp_source") or "environment",
        reward=extra.get("reward"),
    )


def _tool_call_to_adp_action(step: Step, tool_call: ToolCall) -> ApiAction | CodeAction:
    extra = tool_call.extra or {}
    description = content_to_text(step.message)
    reward = (step.extra or {}).get("reward")
    if extra.get("adp_class") == "code_action":
        language = extra.get("language") or tool_call.arguments.get("language") or "python"
        content = (
            tool_call.arguments.get("command")
            or tool_call.arguments.get("code")
            or tool_call.arguments.get("content")
            or ""
        )
        return CodeAction(
            tool_call_id=tool_call.tool_call_id,
            language=language,
            content=content,
            description=description or None,
            reasoning_content=step.reasoning_content,
            reward=reward,
        )
    return ApiAction(
        tool_call_id=tool_call.tool_call_id,
        function=tool_call.function_name,
        kwargs=tool_call.arguments,
        description=description or None,
        reasoning_content=step.reasoning_content,
        reward=reward,
    )


def atif_trajectory_to_adp(trajectory: ATIFTrajectory) -> ADPTrajectory:
    content: list[Any] = []
    details = {
        "atif_schema_version": trajectory.schema_version,
        "atif_agent": trajectory.agent.model_dump(exclude_none=True),
    }
    if trajectory.extra:
        details.update(trajectory.extra)

    for step in trajectory.steps:
        if step.source == "system":
            content.append(
                TextObservation(
                    content=content_to_text(step.message), source="environment", name="system"
                )
            )
            continue
        if step.source == "user":
            content.append(TextObservation(content=content_to_text(step.message), source="user"))
            continue

        if step.tool_calls:
            results_by_call_id = {
                result.source_call_id: result
                for result in (step.observation.results if step.observation else [])
                if result.source_call_id is not None
            }
            for tool_call in step.tool_calls:
                content.append(_tool_call_to_adp_action(step, tool_call))
                result = results_by_call_id.get(tool_call.tool_call_id)
                if result is not None:
                    content.append(_observation_from_atif_result(result))
            continue

        message = content_to_text(step.message)
        if step.observation and step.observation.results:
            if message:
                content.append(
                    MessageAction(
                        content=message,
                        description=(step.extra or {}).get("description"),
                        reasoning_content=step.reasoning_content,
                        reward=(step.extra or {}).get("reward"),
                    )
                )
            for result in step.observation.results:
                content.append(_observation_from_atif_result(result))
            continue

        content.append(
            MessageAction(
                content=message,
                description=(step.extra or {}).get("description"),
                reasoning_content=step.reasoning_content,
                reward=(step.extra or {}).get("reward"),
            )
        )

    available_apis = (trajectory.extra or {}).get("adp_available_apis")
    return cast(
        ADPTrajectory,
        create_trajectory_with_tool_call_links(
            id=trajectory.trajectory_id or trajectory.session_id or "atif-trajectory",
            content=content,
            available_apis=available_apis,
            details=details,
        ),
    )


def normalize_atif_trajectory(trajectory: ATIFTrajectory) -> ATIFTrajectory:
    normalized = cast(ATIFTrajectory, trajectory.model_copy(deep=True))
    for step in normalized.steps:
        for tool_call in step.tool_calls or []:
            function_name = tool_call.function_name
            arguments = dict(tool_call.arguments)
            lower_name = function_name.lower()
            normalized_language = None
            if lower_name in {"bash", "shell", "sh"}:
                normalized_language = "bash"
                tool_call.function_name = "execute_bash"
                if "command" not in arguments:
                    arguments = {"command": arguments.get("code") or arguments.get("content") or ""}
            elif lower_name in {"execute", "python", "py", "python3", "ipython"}:
                normalized_language = "python"
                tool_call.function_name = "execute_ipython_cell"
                if "code" not in arguments:
                    arguments = {"code": arguments.get("command") or arguments.get("content") or ""}
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
            if normalized_language and tool_call.extra and "language" in tool_call.extra:
                tool_call.extra = {**tool_call.extra, "language": normalized_language}
            tool_call.arguments = arguments
    return normalized
