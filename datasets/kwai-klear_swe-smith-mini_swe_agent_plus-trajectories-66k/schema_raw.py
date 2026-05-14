from typing import List

from pydantic import BaseModel


class Message(BaseModel):
    content: str
    role: str


class SchemaRaw(BaseModel):
    instance_id: str
    messages: List[Message]
    id: str
