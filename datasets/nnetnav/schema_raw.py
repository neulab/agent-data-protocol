from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str
    content: str

class SchemaRaw(BaseModel):
    dataset: str
    id: str
    messages: List[Message]