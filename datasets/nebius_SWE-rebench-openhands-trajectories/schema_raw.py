from typing import Any, List, Optional, Union

from pydantic import BaseModel


class Function(BaseModel):
    name: str
    arguments: Optional[Union[str, dict[str, Any]]] = None


class ToolCall(BaseModel):
    function: Function
    type: str
    id: str
    index: Optional[int] = None


class Message(BaseModel):
    content: Optional[str] = None
    name: Optional[str] = None
    role: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class SchemaRaw(BaseModel):
    trajectory_id: str
    instance_id: str
    repo: str
    trajectory: List[Message]
    tools: List[dict[str, Any]]
    model_patch: Optional[str] = None
    exit_status: Optional[str] = None
    resolved: Union[bool, int]
    gen_tests_correct: Optional[float] = None
    pred_passes_gen_tests: Optional[float] = None
