from typing import Literal

from pydantic import Field

from schema.observation.observation import Observation


class TextObservation(Observation):
    class_: str = Field("text_observation", description="The class of the observation")
    content: str = Field(..., description="A textual observation")
    source: Literal["user_msg", "agent", "environment"] = Field(
        ..., description="The source of the observation."
    )
