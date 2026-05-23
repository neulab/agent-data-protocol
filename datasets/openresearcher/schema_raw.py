from typing import Any, Optional

from pydantic import BaseModel


class MessageContent(BaseModel):
    channel_config: Optional[dict[str, Any]] = None
    conversation_start_date: Optional[str] = None
    knowledge_cutoff: Optional[str] = None
    model_identity: Optional[str] = None
    reasoning_effort: Optional[str] = None
    text: Optional[str] = None
    tools: Optional[dict[str, Any]] = None
    type: Optional[str] = None


class Message(BaseModel):
    channel: Optional[str] = None
    content: list[MessageContent]
    content_type: Optional[str] = None
    name: Optional[str] = None
    recipient: Optional[str] = None
    role: str


class SchemaRaw(BaseModel):
    qid: int
    question: str
    answer: Optional[str] = None
    messages: list[Message]
    latency_s: Optional[float] = None
    error: Optional[str] = None
    attempts: int
    status: str
    chunk_idx: int
    num_chunks: int
    config: Optional[str] = None
    split: Optional[str] = None
    row_idx: Optional[int] = None
