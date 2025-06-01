from typing import List

from pydantic import BaseModel


class Message(BaseModel):
    content: str
    role: str


class SchemaRaw(BaseModel):
    messages: List[Message]
    instance_id: str
