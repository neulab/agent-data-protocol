from pydantic import BaseModel, ConfigDict, Field


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_call_id: str | None = Field(
        None,
        description=(
            "Stable identifier for the action/tool call that produced this observation. "
            "This must match a preceding Action.tool_call_id when populated."
        ),
        exclude_if=lambda value: value is None,
    )
    reward: float | None = Field(
        None,
        description="Per-step reward signal associated with this observation. "
        "Used for reinforcement learning training data.",
        exclude_if=lambda value: value is None,
    )
