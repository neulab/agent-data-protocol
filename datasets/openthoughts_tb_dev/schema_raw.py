from typing import List, Optional

from pydantic import BaseModel


class FileRecord(BaseModel):
    path: str
    content: str


class SchemaRaw(BaseModel):
    id: str
    instruction: str
    task_toml: str
    solution: Optional[FileRecord] = None
    dockerfile: Optional[FileRecord] = None
    verification_files: List[FileRecord] = []
    source_files: List[str] = []
