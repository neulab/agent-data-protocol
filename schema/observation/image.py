from typing import Literal

from pydantic import BaseModel, Field

from schema.observation.observation import Observation


class BoundingBox(BaseModel):
    x: float = Field(..., description="The x position of the bounding box")
    y: float = Field(..., description="The y position of the bounding box")
    width: float = Field(..., description="The width of the bounding box")
    height: float = Field(..., description="The height of the bounding box")


class ImageAnnotation(BaseModel):
    text: str = Field(..., description="The text annotation")
    element_type: str = Field(..., description="The type of element")
    bounding_box: BoundingBox = Field(..., description="The boxes of the annotation")


class ImageObservation(Observation):
    class_: str = Field("image_observation", description="The class of the observation")
    content: str = Field(..., description="A path to the image content")
    annotations: list[ImageAnnotation] | None = Field(
        None, description="The annotations of the image"
    )
    source: Literal["user_msg", "agent", "environment"] = Field(
        ..., description="The source of the observation"
    )
