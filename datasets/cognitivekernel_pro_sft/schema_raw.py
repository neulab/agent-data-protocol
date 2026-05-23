from typing import List, Literal

from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SchemaRaw(BaseModel):
    id: str
    source_file: str
    source_index: int
    messages: List[Message]
