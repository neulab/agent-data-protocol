from typing import List
from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class SchemaRaw(BaseModel):
    messages: List[Message]
    instance_id: str
    exp_name: str
    fail: bool
