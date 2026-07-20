from typing import Any, List, Optional

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class SchemaRaw(BaseModel):
    """One CRMArena-Pro ReAct rollout: a message list plus task metadata.

    ``messages`` holds the agent trajectory (the ``traj`` field from CRMArena-Pro
    result logs, renamed by ``extract_raw.py``). Assistant messages are plain
    text carrying ``<thought>`` reasoning and a ``<execute>`` or ``<respond>``
    action; ``user`` messages are either the task query, an execute observation
    (prefixed ``Salesforce instance output:``), or a simulated user turn. The
    remaining fields carry the original task context, the org (b2b/b2c), whether
    the rollout was interactive, and the evaluation reward through to ATIF
    ``extra``.
    """

    messages: List[Message]
    id: Optional[str] = None
    org_type: Optional[str] = None
    interactive: Optional[bool] = None
    task_id: Optional[int] = None
    task_type: Optional[str] = None
    gt_answer: Optional[Any] = None
    reward: Optional[Any] = None
    agent_info: Optional[Any] = None
