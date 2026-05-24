from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ToolCallFunction(BaseModel):
    name: str
    arguments: Any = Field(default_factory=dict)


class ToolCall(BaseModel):
    function: ToolCallFunction
    type: str = "function"
    id: Optional[str] = None
    index: Optional[int] = None


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    function_call: Optional[Any] = None


class Tool(BaseModel):
    type: Optional[str] = None
    function: dict[str, Any] = Field(default_factory=dict)


class SchemaRaw(BaseModel):
    id: str
    source_file: str
    row_index: int
    conversations: List[Message]
    tools: List[Tool] = Field(default_factory=list)
