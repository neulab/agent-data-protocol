from typing import List, Optional

from pydantic import BaseModel


class Message(BaseModel):
    content: str
    loss_mask: Optional[int] = None
    role: str


class SchemaRaw(BaseModel):
    id: str
    messages: List[Message]
    data_source: str
    system: Optional[str] = None
