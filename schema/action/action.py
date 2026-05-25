from pydantic import BaseModel, ConfigDict, Field


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_call_id: str | None = Field(
        None,
        description=(
            "Stable identifier for this tool/action call. When populated, exactly "
            "one later observation must use the same tool_call_id so converters can "
            "emit matched tool-call/result pairs."
        ),
        exclude_if=lambda value: value is None,
    )
    reasoning_content: str | None = Field(
        None,
        description="Extended chain-of-thought reasoning or internal thinking from the agent. "
        "This captures deliberate reasoning processes (e.g., <think> blocks) that are separate "
        "from the action's brief description. Aligns with Harbor ATIF's reasoning_content field "
        "and Agent Client Protocol's agent_thought_chunk concept.",
    )
    reward: float | None = Field(
        None,
        description="Per-step reward signal associated with this action. "
        "Used for reinforcement learning training data.",
    )
