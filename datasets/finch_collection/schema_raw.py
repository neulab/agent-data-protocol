from typing import Optional

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class SchemaRaw(BaseModel):
    id: str
    domain: str
    task: str
    trajectory: list[Message]
    improvement_label: str
    improvement_delta: str
    global_uid: str
    instance_uid: str
    iteration: int
    island_id: int
    generation: int
    parent_id: str
    child_id: str
    parent_metrics: str
    child_metrics: str
    parent_code: str
    child_code: str
    metadata: str
    system_prompt: str
    user_prompt: str
    reasoning: Optional[str] = None
    response: str
    system_token_length: int
    user_token_length: int
    reasoning_token_length: int
    response_token_length: int
    source_dataset: str
    source_file: str
    row_index: int

    model_config = {"extra": "allow"}
