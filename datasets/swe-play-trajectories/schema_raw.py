"""Raw data schema for SWE-Play-trajectories dataset."""

from typing import List, Literal, Optional

from pydantic import BaseModel


class Message(BaseModel):
    """A single message in the conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class SchemaRaw(BaseModel):
    """Raw data schema for SWE-Play-trajectories dataset."""

    id: str
    messages: List[Message]
    category: Optional[str] = None
    source: Optional[str] = None
