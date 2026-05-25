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

    @model_validator(mode="after")
    def validate_tool_call_links(self):
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

        return self
