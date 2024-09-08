from pydantic import BaseModel
from typing import List


class NegCandidate(BaseModel):
    attributes: str
    backend_node_id: str
    tag: str


class Operation(BaseModel):
    op: str
    original_op: str
    value: str


class PosCandidate(BaseModel):
    attributes: str
    backend_node_id: str
    is_original_target: bool
    is_top_level_target: bool
    tag: str


class Action(BaseModel):
    action_uid: str
    cleaned_html: str
    neg_candidates: List[NegCandidate]
    operation: Operation
    pos_candidates: List[PosCandidate]
    raw_html: str


class SchemaRaw(BaseModel):
    website: str
    domain: str
    confirmed_task: str
    annotation_id: str
    action_reprs: List[str]
    actions: List[Action]
    subdomain: str
