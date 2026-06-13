"""Helpers for porting legacy standardized dataset converters to ATIF."""

from __future__ import annotations

from typing import Any

from schema.atif import (
    ATIF_SCHEMA_VERSION,
    Agent,
    ATIFObservation,
    ATIFTrajectory,
    ContentPart,
    ImageSource,
    ObservationResult,
    Step,
    ToolCall,
)
from scripts.raw_to_atif_common import renumber_steps


def text_step(content: str, *, source: str = "user") -> Step:
    return Step(step_id=0, source=source, message=content)


def image_content(path: str, annotations: list[dict[str, Any]] | None = None) -> list[ContentPart]:
    extra = {"annotations": annotations} if annotations else None
    return [ContentPart(type="image", source=ImageSource(path=path), extra=extra)]


def text_observation_step(content: str, *, source: str = "environment") -> Step:
    return Step(
        step_id=0,
        source="agent",
        message="",
        observation=ATIFObservation(
            results=[
                ObservationResult(
                    content=content,
                    extra={"source": source},
                )
            ]
        ),
        llm_call_count=0,
    )


def image_observation_step(
    path: str,
    *,
    annotations: list[dict[str, Any]] | None = None,
    source: str = "environment",
) -> Step:
    return Step(
        step_id=0,
        source="agent",
        message="",
        observation=ATIFObservation(
            results=[
                ObservationResult(
                    content=image_content(path, annotations),
                    extra={"source": source},
                )
            ]
        ),
        llm_call_count=0,
    )


def web_observation_step(
    *,
    html: str | None = None,
    axtree: str | None = None,
    url: str | None = None,
    image_path: str | None = None,
    image_annotations: list[dict[str, Any]] | None = None,
    viewport_size: list[int] | tuple[int, int] | None = None,
) -> Step:
    image_observation = None
    if image_path is not None:
        image_observation = {
            "content": image_path,
            "source": "environment",
            "annotations": image_annotations or [],
        }
    return Step(
        step_id=0,
        source="agent",
        message="",
        observation=ATIFObservation(
            results=[
                ObservationResult(
                    content=html or axtree or url or "",
                    extra={
                        "source": "environment",
                        "web": {
                            "html": html,
                            "axtree": axtree,
                            "url": url,
                            "image_observation": image_observation,
                            "viewport_size": list(viewport_size) if viewport_size else None,
                        },
                    },
                )
            ]
        ),
        llm_call_count=0,
    )


def tool_step(
    function_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    message: str = "",
    description: str | None = None,
    observation: ObservationResult | None = None,
) -> Step:
    call = ToolCall(
        tool_call_id="call_1",
        function_name=function_name,
        arguments=arguments or {},
    )
    return Step(
        step_id=0,
        source="agent",
        message=message,
        tool_calls=[call],
        observation=ATIFObservation(results=[observation]) if observation else None,
        extra={"description": description} if description else None,
    )


def trajectory(
    dataset_name: str,
    trajectory_id: str,
    steps: list[Step],
    *,
    raw: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
    source_agent: str | None = None,
) -> ATIFTrajectory:
    extra: dict[str, Any] = {"source_dataset": dataset_name}
    if raw is not None:
        extra["raw"] = raw
    if details:
        extra.update(details)
    return ATIFTrajectory(
        schema_version=ATIF_SCHEMA_VERSION,
        session_id=trajectory_id,
        trajectory_id=trajectory_id,
        agent=Agent(name=source_agent or dataset_name, version="raw"),
        steps=renumber_steps(steps),
        extra=extra,
    )
