from pydantic import BaseModel


class SchemaRaw(BaseModel):
    timestamp: int
    instance_id: str
    exit_status: str
    stitched: str
    full: str
    result: str
    resolved: bool
