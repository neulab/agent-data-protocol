from typing import List

from pydantic import BaseModel


class OCR(BaseModel):
    text: List[str]
    value: List[List[float]]


class Color(BaseModel):
    text: List[str]
    value: List[List[float]]


class Icon(BaseModel):
    text: List[str]
    value: List[List[float]]


class Box(BaseModel):
    top_left: List[List[float]]
    bottom_right: List[List[float]]
    label: List[str]


class SchemaRaw(BaseModel):
    id: str
    task: str
    image: str
    ocr: OCR
    color: Color
    icon: Icon
    box: Box
