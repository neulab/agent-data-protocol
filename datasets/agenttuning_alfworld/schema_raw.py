from pydantic import BaseModel
from typing import List


class Conversation(BaseModel):
    role: str
    content: str


class SchemaRaw(BaseModel):
    id: str
    conversations: List[Conversation]
