from typing import List

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class SchemaRaw(BaseModel):
    instance_id: str
    repo: str
    messages: List[Message]
    trajectory_format: str
    exit_status: str
    duration_sec: float
