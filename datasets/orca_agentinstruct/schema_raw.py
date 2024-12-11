from pydantic import BaseModel
from typing import List

class Conversation(BaseModel):
    content: str
    role: str

class SchemaRaw(BaseModel):
    id: str
    conversations: List[Conversation]