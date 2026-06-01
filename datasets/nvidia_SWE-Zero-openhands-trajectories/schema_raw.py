from typing import Any, List, Optional, Union

from pydantic import BaseModel


class Function(BaseModel):
    name: str
    arguments: Optional[Union[str, dict[str, Any]]] = None


class ToolCall(BaseModel):
    function: Function
    type: str
    id: str


class Message(BaseModel):
    content: Optional[str] = None
    role: str
    tool_calls: Optional[List[ToolCall]] = None


class SchemaRaw(BaseModel):
    instance_id: str
    repo: str
    license: str
    trajectory_id: str
    trajectory: List[Message]
    model_patch: str
    dataset: str
