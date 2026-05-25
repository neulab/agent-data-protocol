from typing import Literal

from pydantic import Field, field_validator, model_validator

from schema.observation.observation import Observation


class TextObservation(Observation):
    class_: str = Field("text_observation", description="The class of the observation")
    content: str = Field(..., description="A textual observation")
    name: str | None = Field(None, description="An optional name for the participant")
    source: Literal["user", "agent", "environment"] = Field(
        ..., description="The source of the observation."
    )

    @field_validator("class_")
    def validate_class(cls, v):
        if v != "text_observation":
            raise ValueError(f"class_ must be 'text_observation', got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_tool_result_source(self):
        if self.tool_call_id is not None and self.source == "user":
            raise ValueError(
                "TextObservation with tool_call_id represents a tool result and "
                "must not use source='user'"
            )
        return self
