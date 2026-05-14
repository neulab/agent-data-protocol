from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict


class CacheControl(BaseModel):
    type: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class ContentBlock(BaseModel):
    type: Optional[str] = None
    text: Optional[str] = None
    cache_control: Optional[CacheControl] = None

    model_config = ConfigDict(extra="allow")


class Function(BaseModel):
    name: str
    arguments: str = "{}"

    model_config = ConfigDict(extra="allow")


class ToolCall(BaseModel):
    id: Optional[str] = None
    type: Optional[str] = None
    function: Function

    model_config = ConfigDict(extra="allow")


class Message(BaseModel):
    content: Optional[Union[str, List[ContentBlock]]] = None
    role: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    cache_control: Optional[CacheControl] = None

    model_config = ConfigDict(extra="allow")


class ChatMessages(BaseModel):
    messages: List[Message]
    tools: Optional[List[dict[str, Any]]] = None

    model_config = ConfigDict(extra="allow")


class RewardDict(BaseModel):
    multilevel_localization_f1_reward: Optional[float] = None
    file_reward: Optional[float] = None
    module_reward: Optional[float] = None
    entity_reward: Optional[float] = None
    multiturn_reward: Optional[float] = None

    model_config = ConfigDict(extra="allow")


class SchemaRaw(BaseModel):
    source_dataset: str
    source_config: Optional[str] = None
    source_split: str = "train"
    row_id: int
    instance_id: Optional[str] = None
    reward_dict: Optional[RewardDict] = None
    chat_messages: Optional[ChatMessages] = None
    messages: Optional[List[Message]] = None
    step: Optional[int] = None
    rollout_number: Optional[int] = None

    model_config = ConfigDict(extra="allow")
