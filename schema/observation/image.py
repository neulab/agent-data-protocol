from typing import Literal

from pydantic import BaseModel, Field, field_validator

from schema.observation.observation import Observation


class BoundingBox(BaseModel):
    x: float = Field(..., description="The x position of the bounding box")
    y: float = Field(..., description="The y position of the bounding box")
    width: float = Field(..., description="The width of the bounding box")
    height: float = Field(..., description="The height of the bounding box")


class ImageAnnotation(BaseModel):
    element_type: str = Field(..., description="The type of element")
    bounding_box: BoundingBox = Field(..., description="The boxes of the annotation")
    text: str | None = Field(..., description="The exact text annotation (OCR text)")
    content_description: str | None = Field(
        None, description="A description of the content of the element"
    )
    clickable: bool | None = Field(
        None, description="Whether the element can be clicked/tapped"
    )
    editable: bool | None = Field(
        None, description="Whether the element can be edited (text input)"
    )


class ImageObservation(Observation):
    class_: str = Field("image_observation", description="The class of the observation")
    content: str = Field(..., description="A path to the image content")
    annotations: list[ImageAnnotation] | None = Field(
        None, description="The annotations of the image"
    )
    source: Literal["user", "agent", "environment"] = Field(
        ..., description="The source of the observation"
    )

    @field_validator("class_")
    def validate_class(cls, v):
        if v != "image_observation":
            raise ValueError(f"class_ must be 'image_observation', got '{v}'")
        return v
