from typing import List

from pydantic import BaseModel, ConfigDict, Field


class Conversation(BaseModel):
    from_: str = Field(alias="from")
    value: str
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class SchemaRaw(BaseModel):
    id: str
    conversations: List[Conversation]
    model_config = ConfigDict(extra="allow")
