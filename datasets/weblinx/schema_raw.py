from typing import Any, List, Optional
from pydantic import BaseModel


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
    description: str
    status: str
    tasks: List[str]
    data: List[Step]