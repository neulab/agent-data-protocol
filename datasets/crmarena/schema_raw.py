from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ToolCallFunction(BaseModel):
    name: str
    arguments: Any = Field(default_factory=dict)


class ToolCall(BaseModel):
    type: str = "function"
    function: ToolCallFunction
    id: Optional[str] = None


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class SchemaRaw(BaseModel):
    """One CRMArena rollout: an OpenAI-style message list plus task metadata.

    ``messages`` holds the agent trajectory (the ``traj`` field from CRMArena
    result logs, renamed by ``extract_raw.py``). The remaining fields carry the
    original task context and evaluation reward through to ATIF ``extra``.
    """

    messages: List[Message]
    id: Optional[str] = None
    task_id: Optional[int] = None
    task_type: Optional[str] = None
    gt_answer: Optional[str] = None
    reward: Optional[Any] = None
    agent_info: Optional[Any] = None
