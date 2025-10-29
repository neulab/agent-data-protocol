from typing import List

from pydantic import BaseModel


class Message(BaseModel):
    content: str
    role: str


class SchemaRaw(BaseModel):
    messages: List[Message]
    patch: str
    model: str
    instance_id: str
    verified: bool
    id: str
