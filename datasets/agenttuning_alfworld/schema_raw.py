from typing import List

from pydantic import BaseModel


class Conversation(BaseModel):
    role: str
    content: str


class SchemaRaw(BaseModel):
    id: str
    conversations: List[Conversation]
