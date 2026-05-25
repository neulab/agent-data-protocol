from typing import Any, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from schema.action.action import Action
from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.image import ImageObservation
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.version import SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS


class Trajectory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="ADP standardized schema version used by this trajectory.",
    )
    id: str
    content: list[
        Union[
            ApiAction,
            CodeAction,
            MessageAction,
            TextObservation,
            ImageObservation,
            WebObservation,
        ]
    ]
    available_apis: list[str] | None = Field(
        default=None,
        description=(
            "API function names available to this trajectory. Only populate this for "
            "datasets with api.py where the source data explicitly specifies per-instance "
            "tool availability. When provided, this must be a subset of the dataset's api.py "
            "functions and must include every ApiAction function used in the trajectory."
        ),
        exclude_if=lambda value: value is None,
    )

    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details about the trajectory that vary by dataset",
    )

    @field_validator("schema_version")
    def validate_schema_version(cls, value):
        if value not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported schema_version {value!r}. "
                f"Supported versions: {SUPPORTED_SCHEMA_VERSIONS}"
            )
        return value

    @field_validator("content")
    def validate_content_has_class(cls, content):
        for item in content:
            if not hasattr(item, "class_"):
                raise ValueError(
                    f"All content items must have a 'class_' field. Found item: {item}"
                )
        return content

    def _next_generated_tool_call_id(self, existing_ids: set[str], ordinal: int) -> str:
        while True:
            tool_call_id = f"call_{ordinal:06d}"
            if tool_call_id not in existing_ids:
                existing_ids.add(tool_call_id)
                return tool_call_id
            ordinal += 1

    def _backfill_adjacent_tool_call_links(self):
        existing_ids = {
            tool_call_id
            for item in self.content
            if (tool_call_id := getattr(item, "tool_call_id", None)) is not None
        }
        generated_ordinal = 1

        for index, item in enumerate(self.content[:-1]):
            if not isinstance(item, (ApiAction, CodeAction)):
                continue

            next_item = self.content[index + 1]
            if not isinstance(next_item, Observation):
                continue

            action_tool_call_id = item.tool_call_id
            observation_tool_call_id = next_item.tool_call_id
            if action_tool_call_id is not None and observation_tool_call_id is not None:
                continue

            if action_tool_call_id is None and observation_tool_call_id is None:
                action_tool_call_id = self._next_generated_tool_call_id(
                    existing_ids, generated_ordinal
                )
                generated_ordinal += 1
            elif action_tool_call_id is None:
                action_tool_call_id = observation_tool_call_id
            else:
                existing_ids.add(action_tool_call_id)

            item.tool_call_id = action_tool_call_id
            next_item.tool_call_id = action_tool_call_id
            if (
                isinstance(next_item, (TextObservation, ImageObservation))
                and next_item.source == "user"
            ):
                next_item.source = "environment"

    @model_validator(mode="after")
    def validate_tool_call_links(self):
        self._backfill_adjacent_tool_call_links()

        action_indices: dict[str, int] = {}
        matched_observation_indices: dict[str, int] = {}

        for index, item in enumerate(self.content):
            tool_call_id = getattr(item, "tool_call_id", None)
            if tool_call_id is None:
                continue

            if isinstance(item, Action):
                if tool_call_id in action_indices:
                    raise ValueError(
                        f"Duplicate Action.tool_call_id {tool_call_id!r} at content "
                        f"indices {action_indices[tool_call_id]} and {index}"
                    )
                action_indices[tool_call_id] = index
                continue

            if isinstance(item, Observation):
                if tool_call_id not in action_indices:
                    raise ValueError(
                        f"Observation.tool_call_id {tool_call_id!r} at content index "
                        f"{index} does not match any preceding Action.tool_call_id"
                    )
                if tool_call_id in matched_observation_indices:
                    raise ValueError(
                        f"Duplicate observation result for tool_call_id {tool_call_id!r} "
                        f"at content indices {matched_observation_indices[tool_call_id]} "
                        f"and {index}"
                    )
                matched_observation_indices[tool_call_id] = index

        for tool_call_id, action_index in action_indices.items():
            if tool_call_id not in matched_observation_indices:
                raise ValueError(
                    f"Action.tool_call_id {tool_call_id!r} at content index "
                    f"{action_index} does not have a matching Observation.tool_call_id"
                )

        return self
