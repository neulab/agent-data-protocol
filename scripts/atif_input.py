"""Input helpers for SFT converters that consume ATIF trajectories."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from schema.atif import (
    ATIFTrajectory,
    ContentPart,
    ObservationResult,
    ToolCall,
    content_to_text,
    normalize_atif_trajectory,
)


@dataclass
class ApiAction:
    function: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    description: str | None = None
    reasoning_content: str | None = None
    reward: Any | None = None
    tool_call_id: str | None = None


@dataclass
class CodeAction:
    language: str
    content: str
    description: str | None = None
    reasoning_content: str | None = None
    reward: Any | None = None
    tool_call_id: str | None = None


@dataclass
class MessageAction:
    content: str
    description: str | None = None
    reasoning_content: str | None = None
    reward: Any | None = None


@dataclass
class TextObservation:
    content: str
    source: str = "environment"
    tool_call_id: str | None = None
    name: str | None = None
    reward: Any | None = None


@dataclass
class ImageAnnotation:
    text: str | None = None
    element_type: str | None = None
    bbox: Any | None = None


@dataclass
class ImageObservation:
    content: str
    source: str = "environment"
    annotations: list[ImageAnnotation] | None = None
    tool_call_id: str | None = None
    reward: Any | None = None


@dataclass
class WebObservation:
    html: str | None = None
    axtree: str | None = None
    url: str | None = None
    image_observation: ImageObservation | None = None
    viewport_size: tuple[int, int] | None = None
    source: str = "environment"
    tool_call_id: str | None = None
    reward: Any | None = None


@dataclass
class Trajectory:
    id: str
    content: list[Any]
    available_apis: list[str] | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def available_custom_tools(self) -> list[str] | None:
        return self.available_apis


def _description(step_message: Any) -> str | None:
    text = content_to_text(step_message).strip()
    return text or None


def _reward(extra: dict[str, Any] | None) -> Any | None:
    return (extra or {}).get("reward")


def _code_action_from_tool_call(
    tool_call: ToolCall,
    *,
    description: str | None,
    reasoning_content: str | None,
    reward: Any | None,
) -> CodeAction | None:
    name = tool_call.function_name
    args = tool_call.arguments
    if name == "execute_bash":
        return CodeAction(
            tool_call_id=tool_call.tool_call_id,
            language="bash",
            content=str(args.get("command", "")),
            description=description,
            reasoning_content=reasoning_content,
            reward=reward,
        )
    if name == "execute_ipython_cell":
        return CodeAction(
            tool_call_id=tool_call.tool_call_id,
            language="python",
            content=str(args.get("code", "")),
            description=description,
            reasoning_content=reasoning_content,
            reward=reward,
        )
    if name == "execute_code":
        language = str(args.get("language") or (tool_call.extra or {}).get("language") or "python")
        content = args.get("content", args.get("code", args.get("command", "")))
        return CodeAction(
            tool_call_id=tool_call.tool_call_id,
            language=language,
            content=str(content),
            description=description,
            reasoning_content=reasoning_content,
            reward=reward,
        )
    return None


def _action_from_tool_call(step_message: Any, reasoning_content: str | None, tool_call: ToolCall):
    description = _description(step_message)
    reward = _reward(tool_call.extra)
    code_action = _code_action_from_tool_call(
        tool_call,
        description=description,
        reasoning_content=reasoning_content,
        reward=reward,
    )
    if code_action is not None:
        return code_action
    return ApiAction(
        tool_call_id=tool_call.tool_call_id,
        function=tool_call.function_name,
        kwargs=tool_call.arguments,
        description=description,
        reasoning_content=reasoning_content,
        reward=reward,
    )


def _annotations(raw_annotations: Any) -> list[ImageAnnotation] | None:
    if not isinstance(raw_annotations, list):
        return None
    annotations = []
    for item in raw_annotations:
        if isinstance(item, dict):
            annotations.append(
                ImageAnnotation(
                    text=item.get("text"),
                    element_type=item.get("element_type"),
                    bbox=item.get("bbox"),
                )
            )
    return annotations or None


def _image_observation_from_part(
    part: ContentPart,
    *,
    source: str,
    tool_call_id: str | None,
    reward: Any | None,
) -> ImageObservation | None:
    if part.type != "image" or part.source is None:
        return None
    return ImageObservation(
        tool_call_id=tool_call_id,
        content=part.source.path,
        source=source,
        annotations=_annotations((part.extra or {}).get("annotations")),
        reward=reward,
    )


def _observation_from_result(result: ObservationResult):
    extra = result.extra or {}
    source = str(extra.get("source") or extra.get("adp_source") or "environment")
    reward = _reward(extra)
    web = extra.get("web")
    if isinstance(web, dict):
        image = web.get("image_observation")
        image_observation = None
        if isinstance(image, dict):
            image_observation = ImageObservation(
                content=str(image.get("content", "")),
                source=str(image.get("source", "environment")),
                annotations=_annotations(image.get("annotations")),
            )
        viewport = web.get("viewport_size")
        return WebObservation(
            tool_call_id=result.source_call_id,
            html=web.get("html"),
            axtree=web.get("axtree"),
            url=web.get("url"),
            image_observation=image_observation,
            viewport_size=tuple(viewport) if isinstance(viewport, (list, tuple)) else None,
            source=source,
            reward=reward,
        )
    if isinstance(result.content, list):
        for part in result.content:
            image_observation = _image_observation_from_part(
                part,
                source=source,
                tool_call_id=result.source_call_id,
                reward=reward,
            )
            if image_observation is not None:
                return image_observation
    return TextObservation(
        tool_call_id=result.source_call_id,
        content=content_to_text(result.content),
        source=source,
        reward=reward,
    )


def _trajectory_events(trajectory: ATIFTrajectory) -> list[Any]:
    events: list[Any] = []
    for step in trajectory.steps:
        if step.source == "system":
            events.append(
                TextObservation(
                    content=content_to_text(step.message), source="environment", name="system"
                )
            )
            continue
        if step.source == "user":
            events.append(TextObservation(content=content_to_text(step.message), source="user"))
            continue

        results = list(step.observation.results if step.observation else [])
        results_by_call_id = {
            result.source_call_id: result for result in results if result.source_call_id is not None
        }
        consumed_result_ids: set[int] = set()
        if step.tool_calls:
            for tool_call in step.tool_calls:
                events.append(
                    _action_from_tool_call(step.message, step.reasoning_content, tool_call)
                )
                result = results_by_call_id.get(tool_call.tool_call_id)
                if result is not None:
                    consumed_result_ids.add(id(result))
                    events.append(_observation_from_result(result))
            for result in results:
                if id(result) not in consumed_result_ids:
                    events.append(_observation_from_result(result))
            continue

        message = content_to_text(step.message)
        if message:
            events.append(
                MessageAction(
                    content=message,
                    description=(step.extra or {}).get("description"),
                    reasoning_content=step.reasoning_content,
                    reward=_reward(step.extra),
                )
            )
        for result in results:
            events.append(_observation_from_result(result))
    return events


def load_trajectory(line: str) -> Trajectory:
    data: dict[str, Any] = json.loads(line)
    atif_trajectory = normalize_atif_trajectory(ATIFTrajectory(**data))
    trajectory_id = atif_trajectory.trajectory_id or atif_trajectory.session_id or "atif-trajectory"
    extra = atif_trajectory.extra or {}
    return Trajectory(
        id=trajectory_id,
        content=_trajectory_events(atif_trajectory),
        available_apis=extra.get("adp_available_apis"),
        details=extra,
    )
