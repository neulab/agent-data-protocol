from pydantic import BaseModel, ConfigDict


class SchemaRaw(BaseModel):
    messages: str
    id: str
    reward: float
    model_config = ConfigDict(extra="allow")
