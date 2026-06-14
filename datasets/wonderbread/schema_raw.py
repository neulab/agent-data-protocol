from typing import Any

from pydantic import BaseModel, ConfigDict


class StateData(BaseModel):
    id: int
    timestamp: str
    sec_from_start: float
    url: str
    tab: str
    json_state: str
    html: str
    screenshot_base64: str | None
    path_to_screenshot: str
    screen_size: dict[str, int]


class State(BaseModel):
    type: str
    data: StateData


class ActionData(BaseModel):
    type: str
    timestamp: str
    is_right_click: bool
    pressed: bool
    element_attributes: dict[str, Any]


class Action(BaseModel):
    type: str
    data: dict[str, Any]


class WebArena(BaseModel):
    sites: list[str]
    task_id: int
    required_login: bool
    storage_state: str
    start_url: list[str]
    intent_template: str
    instantiation_dict: dict[str, Any]
    intent: str
    require_reset: bool
    eval: dict[str, Any]
    intent_template_id: int

    model_config = ConfigDict(extra="allow")


class SchemaRaw(BaseModel):
    trace: list[State | Action]
    webarena: WebArena | dict[str, Any]
    sop_text: str | None = None
    task_stamp: str | None = None

    model_config = ConfigDict(extra="allow")
