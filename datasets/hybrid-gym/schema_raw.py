from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    model_config = ConfigDict(extra="allow")


class SchemaRaw(BaseModel):
    id: str | None = None
    source_dataset: str | None = None
    task_type: str | None = None
    split: str = "train"
    row_index: int | None = None
    messages: list[Message] = Field(default_factory=list)
    model_config = ConfigDict(extra="allow")
