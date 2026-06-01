from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    model_config = ConfigDict(extra="forbid")


class SchemaRaw(BaseModel):
    id: str
    source_dataset: str
    task_type: str
    split: str = "train"
    row_index: int
    messages: list[Message] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")
