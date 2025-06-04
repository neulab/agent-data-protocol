from typing import Literal

from pydantic import Field, field_validator

from schema.observation.observation import Observation


class TextObservation(Observation):
    class_: str = Field("text_observation", description="The class of the observation")
    content: str = Field(..., description="A textual observation")
    source: Literal["user", "agent", "environment"] = Field(
        ..., description="The source of the observation."
    )

    @field_validator("class_")
    def validate_class(cls, v):
        if v != "text_observation":
            raise ValueError(f"class_ must be 'text_observation', got '{v}'")
        return v
