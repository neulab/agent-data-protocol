from pydantic import BaseModel
from typing import List

class Message(BaseModel):
    content: str
    role: str

class SchemaRaw(BaseModel):
    messages: List[Message]
    instance_id: str