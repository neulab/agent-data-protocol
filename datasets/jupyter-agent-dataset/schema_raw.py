from typing import Any

from pydantic import BaseModel, Field


class MessageFunction(BaseModel):
    name: str
    arguments: dict[str, str | None] = Field(default_factory=dict)


class MessageToolCall(BaseModel):
    function: MessageFunction


class Message(BaseModel):
    content: str | None = None
    role: str
    tool_calls: list[MessageToolCall] | None = None


class ToolDefinition(BaseModel):
    function: dict[str, Any]
    type: str


class SchemaRaw(BaseModel):
    messages: list[Message]
    id: str
    edu_score: int | None = None
    files_used: list[str] | None = None
    packages_used: list[str] | None = None
    question: str
    answer: str
    kaggle_dataset_name: str | None = None
    executor_type: str | None = None
    original_notebook: str | None = None
    tools: list[ToolDefinition] | None = None
    split: str | None = None
