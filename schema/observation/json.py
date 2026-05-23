import json
from typing import Any, Literal

from pydantic import Field, field_validator

from schema.observation.observation import Observation


class JsonObservation(Observation):
    class_: str = Field("json_observation", description="The class of the observation")
    content: dict[str, Any] = Field(..., description="A JSON-serializable observation")
    name: str | None = Field(None, description="An optional name for the participant")
    source: Literal["user", "agent", "environment"] = Field(
        ..., description="The source of the observation."
    )

    @field_validator("class_")
    def validate_class(cls, v):
        if v != "json_observation":
            raise ValueError(f"class_ must be 'json_observation', got '{v}'")
        return v

    @field_validator("content")
    def validate_json_serializable(cls, v):
        try:
            json.dumps(v)
        except TypeError as exc:
            raise ValueError("content must be JSON-serializable") from exc
        return v
