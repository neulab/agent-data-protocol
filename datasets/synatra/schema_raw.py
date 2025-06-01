from pydantic import BaseModel


class SchemaRaw(BaseModel):
    prompt: str
    response: str
