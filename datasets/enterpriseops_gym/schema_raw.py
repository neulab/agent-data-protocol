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
    """EnterpriseOps-Gym raw records are OpenAI-style chat trajectories.

    EnterpriseOps-Gym publishes *tasks* (a system prompt, a user request, the
    oracle tool set, and SQL ``verifiers``) that are evaluated against live,
    resettable MCP servers -- it does NOT publish ready-made agent rollouts.
    A rollout is produced by running an agent against the gym and captured here
    as a list of chat messages (``system``/``user``/``assistant``/``tool``)
    where assistant turns carry MCP tool calls and ``tool`` turns carry the
    environment observations. Extra top-level fields (``task``, ``domain``,
    ``verifiers``, ``reward`` ...) are preserved by the shared converter.
    """

    messages: List[Message]
