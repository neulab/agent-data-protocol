from typing import List, Optional

from pydantic import BaseModel


class ConversationTurn(BaseModel):
    role: str
    content: str


class SchemaRaw(BaseModel):
    conversations: List[ConversationTurn]
    agent: Optional[str] = None
    model: Optional[str] = None
    model_provider: Optional[str] = None
    date: Optional[str] = None
    task: Optional[str] = None
    episode: Optional[str] = None
    run_id: Optional[str] = None
    trial_name: Optional[str] = None
    enable_thinking: Optional[bool] = None
