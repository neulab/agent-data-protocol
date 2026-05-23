from typing import List, Literal, Optional

from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class SchemaRaw(BaseModel):
    id: str
    category: str
    source: str
    messages: List[Message]
    generator: Optional[str] = None
    thinking: Optional[bool] = None
    patch: Optional[str] = None
