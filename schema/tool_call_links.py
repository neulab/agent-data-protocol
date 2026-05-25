from typing import Any

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.observation.image import ImageObservation
from schema.observation.observation import Observation
from schema.observation.text import TextObservation


def _next_generated_tool_call_id(existing_ids: set[str], ordinal: int) -> str:
    while True:
        tool_call_id = f"call_{ordinal:06d}"
        if tool_call_id not in existing_ids:
            existing_ids.add(tool_call_id)
            return tool_call_id
        ordinal += 1


def _normalize_linked_tool_result_source(item: Observation) -> None:
    if isinstance(item, (TextObservation, ImageObservation)) and item.source == "user":
        item.source = "environment"


def backfill_adjacent_tool_call_links(content: list[Any]) -> list[Any]:
    """Add IDs to adjacent tool-action/result pairs in converter output."""
    existing_ids = {
        tool_call_id
        for item in content
        if (tool_call_id := getattr(item, "tool_call_id", None)) is not None
    }
    generated_ordinal = 1

    for index, item in enumerate(content[:-1]):
        if not isinstance(item, (ApiAction, CodeAction)):
            continue

        next_item = content[index + 1]
        if not isinstance(next_item, Observation):
            continue

        action_tool_call_id = item.tool_call_id
        observation_tool_call_id = next_item.tool_call_id
        _normalize_linked_tool_result_source(next_item)
        if action_tool_call_id is not None and observation_tool_call_id is not None:
            continue

        if action_tool_call_id is None and observation_tool_call_id is None:
            action_tool_call_id = _next_generated_tool_call_id(existing_ids, generated_ordinal)
            generated_ordinal += 1
        elif action_tool_call_id is None:
            action_tool_call_id = observation_tool_call_id

        item.tool_call_id = action_tool_call_id
        next_item.tool_call_id = action_tool_call_id

    return content


def infer_available_code_languages(content: list[Any]) -> list[str] | None:
    languages = sorted({str(item.language) for item in content if isinstance(item, CodeAction)})
    return languages or None


def create_trajectory_with_tool_call_links(**kwargs):
    from schema.trajectory import Trajectory

    kwargs["content"] = backfill_adjacent_tool_call_links(kwargs["content"])
    if kwargs.get("available_code_languages") is None:
        inferred_code_languages = infer_available_code_languages(kwargs["content"])
        if inferred_code_languages is not None:
            kwargs["available_code_languages"] = inferred_code_languages
    return Trajectory(**kwargs)
