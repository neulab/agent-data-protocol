from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    model_config = ConfigDict(extra="allow")


class SchemaRaw(BaseModel):
    messages: list[Message]
    split: Optional[str] = None
    id: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class McpToolCall(BaseModel):
    server_name: str
    tool_name: str
    arguments: dict[str, Any] | str
