from typing import List, Optional
from pydantic import BaseModel, Field, root_validator, ConfigDict


class SchemaRaw(BaseModel):
    _id: str
    Task: str
    Title: str
    Description: str
    Keywords: str
    Template: str
    Answer: dict[str, str]

    model_config = ConfigDict(extra="allow")