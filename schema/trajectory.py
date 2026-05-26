from typing import Any, Union, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    available_custom_tools: list[str] | None = Field(
        default=None,
        description=(
            "Custom tool names available to this trajectory. Only populate this when "
            "the source data explicitly specifies per-instance custom tool availability. "
            "When provided, this must be a subset of metadata.json custom_tools and "
            "must include every custom ApiAction.function used in the trajectory."
        ),
        exclude_if=lambda value: value is None,
    )
    available_code_languages: list[str] | None = Field(
        default=None,
        description=(
            "Code action languages available to this trajectory. Populate this when "
            "the trajectory contains CodeAction entries so converters can expose only "
            "the code/executor tools that are used by the trajectory. When provided, "
            "this must exactly match the CodeAction languages used by the trajectory."
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

    @field_validator("available_code_languages")
    def validate_available_code_languages(cls, value):
        if value is None:
            return value
        valid_languages = set(get_args(CodeAction.model_fields["language"].annotation))
        invalid_languages = sorted(set(value) - valid_languages)
        if invalid_languages:
            raise ValueError(
                "available_code_languages contains unsupported CodeAction languages: "
                f"{invalid_languages}"
            )
        return value

    @model_validator(mode="after")
    def validate_tool_call_links(self):
        action_indices: dict[str, int] = {}
        matched_observation_indices: dict[str, int] = {}

        for index, item in enumerate(self.content):
            tool_call_id = getattr(item, "tool_call_id", None)
            next_item = self.content[index + 1] if index + 1 < len(self.content) else None
            if (
                isinstance(item, (ApiAction, CodeAction))
                and isinstance(next_item, Observation)
                and tool_call_id is None
            ):
                raise ValueError(
                    f"Tool action at content index {index} is followed by an "
                    "Observation result but does not include tool_call_id"
                )
            if tool_call_id is None:
                continue

            if isinstance(item, MessageAction):
                raise ValueError(
                    f"MessageAction.tool_call_id {tool_call_id!r} at content index "
                    f"{index} is not allowed because MessageAction is not a tool call"
                )

            if isinstance(item, (ApiAction, CodeAction)):
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

    @model_validator(mode="after")
    def validate_available_code_languages_cover_used_actions(self):
        if self.available_code_languages is None:
            return self
        available_code_languages = set(self.available_code_languages)
        used_code_languages = {
            item.language for item in self.content if isinstance(item, CodeAction)
        }
        missing_languages = sorted(used_code_languages - available_code_languages)
        if missing_languages:
            raise ValueError(
                "CodeAction languages are missing from available_code_languages: "
                f"{missing_languages}"
            )
        unused_languages = sorted(available_code_languages - used_code_languages)
        if unused_languages:
            raise ValueError(
                "available_code_languages contains languages not used by CodeAction "
                f"entries: {unused_languages}"
            )
        return self
