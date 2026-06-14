from pydantic import BaseModel, ConfigDict


class SchemaRaw(BaseModel):
    _id: str
    Task: str
    Title: str
    Description: str
    Keywords: str
    Template: str

    model_config = ConfigDict(extra="allow")
