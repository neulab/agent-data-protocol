from typing import List, Optional

from pydantic import BaseModel


class TrajectoryItem(BaseModel):
    text: Optional[str] = None
    cutoff_date: Optional[str] = None
    mask: Optional[bool] = None
    role: Optional[str] = None
    system_prompt: Optional[str] = None


class SchemaRaw(BaseModel):
    instance_id: Optional[str] = None
    model_name: Optional[str] = None
    target: Optional[bool] = None
    trajectory: List[TrajectoryItem] = []
    exit_status: Optional[str] = None
    generated_patch: Optional[str] = None
    eval_logs: Optional[str] = None
