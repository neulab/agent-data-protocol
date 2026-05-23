from typing import List

from pydantic import BaseModel


class ConversationTurn(BaseModel):
    role: str
    content: str


class SchemaRaw(BaseModel):
    id: str
    conversations: List[ConversationTurn]
