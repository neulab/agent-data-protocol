from pydantic import BaseModel
from typing import List, Optional, Union


class BBoxPixels(BaseModel):
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    center: List[float]
    width: int
    height: int
    # area: int


class AccessibilityTree(BaseModel):
    text: Optional[Union[str, None]] = None
    content_description: Optional[Union[str, None]] = None
    class_name: str
    bbox: Optional[Union[None, None]] = None
    bbox_pixels: BBoxPixels
    hint_text: Optional[Union[str, None]] = None
    is_checked: bool
    is_checkable: bool
    is_clickable: bool
    is_editable: bool
    is_enabled: bool
    is_focused: bool
    is_focusable: bool
    is_long_clickable: bool
    is_scrollable: bool
    is_selected: bool
    is_visible: bool
    # package_name: str
    # resource_name: Optional[Union[str, None]] = None
    tooltip: Optional[Union[None, None]] = None
    # resource_id: Optional[Union[None, None]] = None


class Action(BaseModel):
    action_type: str
    x: Optional[int] = None
    y: Optional[int] = None
    app_name: Optional[str] = None
    direction: Optional[str] = None


class SchemaRaw(BaseModel):
    episode_id: int
    goal: str
    screenshots: List[str]
    accessibility_trees: List[List[AccessibilityTree]]
    screenshot_widths: List[int]
    screenshot_heights: List[int]
    actions: List[Action]
    step_instructions: List[str]
