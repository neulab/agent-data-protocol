from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ToolCallFunction(BaseModel):
    name: str
    arguments: Any = Field(default_factory=dict)


class ToolCall(BaseModel):
    type: str = "function"
    function: ToolCallFunction
    id: Optional[str] = None


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class SchemaRaw(BaseModel):
    messages: List[Message]
