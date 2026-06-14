from typing import List, Literal

from pydantic import BaseModel, ConfigDict


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    model_config = ConfigDict(extra="allow")


class SchemaRaw(BaseModel):
    id: str
    source_file: str
    source_index: int
    messages: List[Message]
    model_config = ConfigDict(extra="allow")
