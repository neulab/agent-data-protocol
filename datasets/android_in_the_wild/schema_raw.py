from typing import List

from pydantic import BaseModel, Field


class SchemaRaw(BaseModel):
    image_width: int = Field(..., alias="image/width")
    image_height: int = Field(..., alias="image/height")
    results_yx_touch: List[float] = Field(..., alias="results/yx_touch")
    results_type_action: str = Field(..., alias="results/type_action")
    android_api_level: int = Field(...)
    image_ui_annotations_text: List[str] = Field(..., alias="image/ui_annotations_text")
    results_yx_lift: List[float] = Field(..., alias="results/yx_lift")
    image_ui_annotations_ui_types: List[str] = Field(..., alias="image/ui_annotations_ui_types")
    current_activity: str = Field(...)
    step_id: int = Field(...)
    image_channels: int = Field(..., alias="image/channels")
    episode_length: int = Field(...)
    image_encoded: str = Field(..., alias="image/encoded")
    results_action_type: str = Field(..., alias="results/action_type")
    goal_info: str = Field(...)
    image_ui_annotations_positions: List[List[float]] = Field(
        ..., alias="image/ui_annotations_positions"
    )
    episode_id: str = Field(...)
    device_type: str = Field(...)
