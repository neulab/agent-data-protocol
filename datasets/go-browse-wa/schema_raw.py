from typing import Any, Dict

from pydantic import BaseModel


class SchemaRaw(BaseModel):
    graph_data: Dict[str, Any]
    node_data: Dict[str, Any]
    step_data: Dict[str, Any]
    traj_data: Dict[str, Any]
