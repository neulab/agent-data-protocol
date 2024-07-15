from typing import Any
from pydantic import BaseModel, Field

class ActionMetadata(BaseModel):
    cot: str

class Action(BaseModel):
    metadata: ActionMetadata
    action: dict[str, Any]

class State(BaseModel):
    url: str
    axtree: str
    screenshot_path: str
    
class SchemaRaw(BaseModel):
    task_id: int
    intent: str
    source: str = Field(..., description="The source model of the trajectory")
    trajectory: list[Action | State]