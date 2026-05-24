from typing import Any, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.image import ImageObservation
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
