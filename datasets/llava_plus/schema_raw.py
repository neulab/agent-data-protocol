from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class APIParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    boxes: list[list[float]] | None = None
    prompt: str | None = None


class Action(BaseModel):
    model_config = ConfigDict(extra="allow")

    API_name: str
    API_params: APIParams


class Conversation(BaseModel):
    model_config = ConfigDict(extra="allow")

    from_: str = Field(..., alias="from")
    value: str
    thoughts: str | None = None
    actions: list[Action] | None = None


class SchemaRaw(BaseModel):
    model_config = ConfigDict(extra="allow")

    unique_id: str | None = None
    id: str | None = None
    image: str | None = None
    conversations: list[Conversation]
    data_source: str | None = None
    metadata: dict[str, Any] | None = None
