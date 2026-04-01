"""Pydantic schema for Mind2Web raw data format.

Source: https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web
Per-action rows are grouped by annotation_id into per-trajectory objects
by extract_raw.py. This schema defines the grouped output format.
"""

from typing import Optional

from pydantic import BaseModel


class Operation(BaseModel):
    """Action type and value for a single step."""

    op: str
    original_op: str
    value: str


class ActionStep(BaseModel):
    """One action step in a trajectory.

    Fields are extracted from Multimodal-Mind2Web per-action rows.
    raw_html and neg_candidates are dropped during extraction.
    """

    action_uid: str
    cleaned_html: str
    operation: Operation
    backend_node_id: Optional[str] = None
    screenshot_path: Optional[str] = None
    action_repr: str

    model_config = {"extra": "allow"}


class SchemaRaw(BaseModel):
    """Root schema for a Mind2Web trajectory.

    Reconstructed from per-action rows in Multimodal-Mind2Web,
    grouped by annotation_id and sorted by target_action_index.
    """

    annotation_id: str
    website: str
    domain: str
    subdomain: str
    confirmed_task: str
    action_reprs: list[str]
    actions: list[ActionStep]

    model_config = {"extra": "allow"}
