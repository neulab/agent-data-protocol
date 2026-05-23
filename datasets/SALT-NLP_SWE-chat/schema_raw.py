from typing import Any

from pydantic import BaseModel, ConfigDict


class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra="allow")

    turn_id: str | None = None
    session_id: str | None = None
    checkpoint_pk: str | None = None
    repo_id: str | None = None
    user_id: str | None = None
    turn_number: int | None = None
    conversation_turn_number: int | None = None
    role: str
    turn_type: str | None = None
    is_conversational: bool | None = None
    content: str | None = None
    model: str | None = None
    timestamp: Any = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    is_continuation: bool | None = None
    is_first_turn: bool | None = None
    word_count: int | None = None
    char_count: int | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    file_path: str | None = None
    command: str | None = None
    pattern: str | None = None
    tool_input_json: Any = None
    category: str | None = None
    bash_category: str | None = None
    queue_op_subtype: str | None = None
    agent: str | None = None
    strategy: str | None = None
    language: str | None = None
    prompt_intent: str | None = None
    prompt_pushback: str | None = None


class SchemaRaw(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_id: str
    repo_id: str | None = None
    checkpoint_pk: str | None = None
    user_id: str | None = None
    agent: str | None = None
    strategy: str | None = None
    branch: str | None = None
    created_at: Any = None
    transcript_path: str | None = None
    tool_call_count: int | None = None
    turn_count: int | None = None
    prompt_count: int | None = None
    agent_percentage: float | None = None
    user_persona: str | None = None
    session_success: str | None = None
    turns: list[ConversationTurn]
