from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class State(BaseModel):
    screenshot: Optional[str] = None
    page: Optional[str] = None
    frame_resized: Optional[bool] = None
    screenshot_status: Optional[str] = None


class Step(BaseModel):
    timestamp: float
    speaker: Optional[str] = None
    utterance: Optional[str] = None
    type: str
    state: Optional[State] = None
    action: Optional[dict[str, Any]] = None


class SchemaRaw(BaseModel):
    shortcode: str
    replay: dict[str, Any]
    form: dict[str, Any]

    model_config = ConfigDict(extra="allow")
