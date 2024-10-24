from pydantic import Field

from schema.observation.image import ImageObservation
from schema.observation.observation import Observation


class WebObservation(Observation):
    html: str | None = Field(..., description="The raw HTML of the web page")
    axtree: str | None = Field(None, description="The Accesibility Tree of the web page")
    url: str | None = Field(..., description="The URL of the web page")
    image_observation: ImageObservation | None = Field(
        ..., description="The image observation"
    )
    viewport_size: tuple[int, int] | None = Field(
        ..., description="The size of the viewport"
    )
