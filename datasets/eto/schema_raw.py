from typing import List

from pydantic import BaseModel, Field


class Conversation(BaseModel):
    from_: str = Field(..., alias="from")
    value: str


class SchemaRaw(BaseModel):
    id: str
    conversations: List[Conversation]
    reward: float
    source: str
