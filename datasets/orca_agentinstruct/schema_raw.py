from pydantic import BaseModel, ConfigDict


class SchemaRaw(BaseModel):
    id: str
    messages: str
    model_config = ConfigDict(extra="allow")
