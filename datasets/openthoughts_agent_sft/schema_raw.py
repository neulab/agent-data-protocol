from typing import Optional

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class SchemaRaw(BaseModel):
    id: str
    conversations: list[Message]
    agent: str
    model: str
    model_provider: str
    date: str
    task: str
    episode: str
    run_id: str
    trial_name: str
    source_dataset: str
    source_file: str
    source_variant: str
    row_index: int
    result: Optional[str] = None
    trace_source: Optional[str] = None
