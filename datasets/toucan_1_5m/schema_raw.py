from typing import Any, Dict, List, Union

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str
    function_call: Union[Dict[str, Any], None] = None


class SchemaRaw(BaseModel):
    id: str
    uuid: str
    subset_name: str
    messages: List[Message]
    question: str
    available_tools: List[Dict[str, Any]]
    target_tools: List[str]
    question_quality_assessment: str
    response_quality_assessment: str
    metadata: Dict[str, Any]
