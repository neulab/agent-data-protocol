from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class Args(BaseModel):
    LLM_MODEL: Optional[str] = None
    AGENT: Optional[str] = None
    LANGUAGE: Optional[str] = None
    LLM_API_KEY: Optional[str] = None
    content: Optional[str] = None
    command: Optional[str] = None
    background: Optional[bool] = None
    thought: Optional[str] = None


class Extras(BaseModel):
    agent_state: Optional[str] = None
    command_id: Optional[int] = None
    command: Optional[str] = None
    exit_code: Optional[int] = None


class TrajectoryItem(BaseModel):
    action: Optional[str] = None
    args: Optional[Args] = None
    token: Optional[str] = None
    status: Optional[str] = None
    id: Optional[int] = None
    timestamp: Optional[datetime] = None
    source: Optional[str] = None
    message: str = ""
    observation: Optional[str] = None
    content: str = ""
    extras: Optional[Extras] = None
    cause: Optional[int] = None


class SchemaRaw(BaseModel):
    version: str
    token: str
    feedback: str
    permissions: str
    trajectory: List[TrajectoryItem]
