from typing import List, Optional
from pydantic import BaseModel


class Action(BaseModel):
    action: str
    args: dict
    command: Optional[str] = None
    background: Optional[bool] = False
    thought: Optional[str] = None


class Cause(BaseModel):
    id: int


class Observation(BaseModel):
    source: str
    message: str
    observation: str
    content: Optional[str] = None
    extras: Optional[dict] = None


class TrajectoryEntry(BaseModel):
    id: int
    timestamp: str
    source: str
    message: str
    action: Optional[str] = None
    args: Optional[dict] = None
    cause: Optional[Cause] = None
    observation: Optional[Observation] = None
    content: Optional[str] = None
    extras: Optional[dict] = None


class LogEntry(BaseModel):
    FAIL_TO_PASS: dict
    PASS_TO_PASS: dict
    FAIL_TO_FAIL: dict
    PASS_TO_FAIL: dict


class Summary(BaseModel):
    repo: str
    total_predictions: int
    Patch_Apply_Success: dict


class Data:
    summary: Summary


class Root(BaseModel):
    version: str
    token: str
    feedback: str
    permissions: str
    trajectory: List[TrajectoryEntry]
    data: Data
