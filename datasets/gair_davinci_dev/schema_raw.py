from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    arguments: Union[str, dict[str, Any]] = "{}"


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    type: Optional[str] = "function"
    function: FunctionCall


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Optional[str] = ""
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class SchemaRaw(BaseModel):
    model_config = ConfigDict(extra="allow")

    messages: list[Message]
    sample_name: Optional[str] = None
    id: Optional[str] = None
