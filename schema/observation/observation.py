from pydantic import BaseModel, ConfigDict, Field


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reward: float | None = Field(
        None,
        description="Per-step reward signal associated with this observation. "
        "Used for reinforcement learning training data.",
    )
