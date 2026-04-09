from pydantic import BaseModel, Field


class Observation(BaseModel):
    reward: float | None = Field(
        None,
        description="Per-step reward signal associated with this observation. "
        "Used for reinforcement learning training data.",
    )
