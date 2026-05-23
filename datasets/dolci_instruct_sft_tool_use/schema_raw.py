from pydantic import BaseModel


class Message(BaseModel):
    content: str | None = None
    function_calls: str | None = None
    functions: str | None = None
    role: str


class SchemaRaw(BaseModel):
    messages: list[Message]
    dataset_source: str
    id: str
