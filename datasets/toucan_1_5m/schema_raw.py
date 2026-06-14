from typing import Any

from pydantic import BaseModel, ConfigDict


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str
    function_call: dict[str, Any] | None = None


class SchemaRaw(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    uuid: str | None = None
    subset_name: str | None = None
    messages: Any
    question: str | None = None
    available_tools: Any = None
    tools: Any = None
    target_tools: Any = None
    question_quality_assessment: str | None = None
    response_quality_assessment: str | None = None
    metadata: Any = None
