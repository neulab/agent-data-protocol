from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    model_config = ConfigDict(extra="allow")


class McpToolDefinition(BaseModel):
    server_name: str
    tool_name: str
    function_name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    argument_name_map: dict[str, str] = Field(default_factory=dict)


class SchemaRaw(BaseModel):
    messages: list[Message]
    split: Optional[str] = None
    id: Optional[str] = None
    available_tools: list[McpToolDefinition] = Field(default_factory=list)
    model_config = ConfigDict(extra="allow")


class McpToolCall(BaseModel):
    server_name: str
    tool_name: str
    arguments: dict[str, Any] | str
