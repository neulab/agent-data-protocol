from typing import List, Optional

from pydantic import BaseModel


class Function(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    function: Function
    type: str
    id: str
    index: Optional[int] = None


class Message(BaseModel):
    content: Optional[str] = None
    function_call: Optional[str] = None
    name: Optional[str] = None
    role: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class SchemaRaw(BaseModel):
    instance_id: str
    run_id: str
    resolved: bool
    messages: List[Message]
    tools: List[dict]
    test_result: dict
