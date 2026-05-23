from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict


class TextContent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    text: str
    cache_control: Optional[dict] = None


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    arguments: str = "{}"
    name: str


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: Optional[int] = None
    function: FunctionCall
    id: Optional[str] = None
    type: str = "function"


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Optional[Union[str, List[TextContent]]] = None
    thought: Optional[str] = None
    action: Optional[str] = None
    agent: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_ids: Optional[List[str]] = None
    message_type: Optional[str] = None
    cache_control: Optional[dict] = None


class SchemaRaw(BaseModel):
    model_config = ConfigDict(extra="allow")

    messages: Union[str, List[Message]]
    instance_id: str
    rollout_patch: str = ""
    func_name: str = ""
    func_path: str = ""
    problem_statement: str = ""
    target_patch: str = ""
    docker_image: str = ""
