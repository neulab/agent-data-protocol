from typing import Optional

from pydantic import BaseModel


class SchemaRaw(BaseModel):
    messages: str
    instance_id: str
    rollout_patch: Optional[str] = None
    func_name: Optional[str] = None
    func_path: Optional[str] = None
    problem_statement: Optional[str] = None
    target_patch: Optional[str] = None
    docker_image: Optional[str] = None
