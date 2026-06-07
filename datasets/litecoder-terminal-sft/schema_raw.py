from typing import Literal

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    from_: Literal["human", "gpt"] = Field(..., alias="from")
    value: str


class SchemaRaw(BaseModel):
    id: int
    conversations: list[ConversationTurn]
