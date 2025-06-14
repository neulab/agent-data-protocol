from typing import List

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class SchemaRaw(BaseModel):
    dataset: str
    id: str
    messages: List[Message]
    n_tokens: int
    prompt: str
    task_name: str
