from pydantic import Field, field_validator

from schema.observation.image import ImageObservation
from schema.observation.observation import Observation


class WebObservation(Observation):
    class_: str = Field("web_observation", description="The class of the observation")
    html: str | None = Field(..., description="The raw HTML of the web page")
    axtree: str | None = Field(None, description="The Accesibility Tree of the web page")
    url: str | None = Field(..., description="The URL of the web page")
    image_observation: ImageObservation | None = Field(..., description="The image observation")
    viewport_size: tuple[int, int] | None = Field(..., description="The size of the viewport")

    @field_validator("class_")
    def validate_class(cls, v):
        if v != "web_observation":
            raise ValueError(f"class_ must be 'web_observation', got '{v}'")
        return v
