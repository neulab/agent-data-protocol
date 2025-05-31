from typing import List

from pydantic import BaseModel


class Conversation(BaseModel):
    content: str
    role: str


class SchemaRaw(BaseModel):
    id: str
    conversations: List[Conversation]
