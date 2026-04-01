"""Pydantic schema for AgentNet raw data format.

Source: https://huggingface.co/datasets/xlangai/AgentNet
Each JSONL line represents one trajectory with PyAutoGUI actions and screenshots.
"""

from typing import Optional

from pydantic import BaseModel


class StepValue(BaseModel):
    """Per-step payload containing observation, reasoning, and action."""

    observation: Optional[str] = None
    thought: Optional[str] = None
    action: Optional[str] = None
    code: Optional[str] = None
    last_step_correct: Optional[bool] = None
    last_step_redundant: Optional[bool] = None
    reflection: Optional[str] = None

    class Config:
        extra = "allow"


class TrajectoryStep(BaseModel):
    """A single step in a trajectory: screenshot + action."""

    index: int
    image: Optional[str] = None
    value: StepValue

    class Config:
        extra = "allow"


class SchemaRaw(BaseModel):
    """Root schema for an AgentNet trajectory.

    Quality scores are per-trajectory (0-10 scale):
    - alignment_score: how well actions aligned with the task objective
    - efficiency_score: how few redundant steps were taken
    - task_difficulty: inherent complexity of the task
    """

    task_id: str
    instruction: Optional[str] = None
    task_completed: Optional[bool] = None
    alignment_score: Optional[int] = None
    efficiency_score: Optional[int] = None
    task_difficulty: Optional[int] = None
    reason: Optional[str] = None
    natural_language_task: Optional[str] = None
    actual_task: Optional[str] = None
    domain: Optional[str] = None
    traj: list[TrajectoryStep]

    class Config:
        extra = "allow"
