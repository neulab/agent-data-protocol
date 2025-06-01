from typing import List, Optional

from pydantic import BaseModel, Field


class APIParams(BaseModel):
    boxes: Optional[List[List[float]]] = None
    prompt: Optional[str] = None


class Action(BaseModel):
    API_name: str
    API_params: APIParams


class Conversation(BaseModel):
    from_: str = Field(..., alias="from")
    value: str
    thoughts: Optional[str] = None
    actions: Optional[List[Action]] = None


class SchemaRaw(BaseModel):
    unique_id: str
    image: str
    conversations: List[Conversation]
    data_source: str
