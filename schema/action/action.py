from pydantic import BaseModel, Field


class Action(BaseModel):
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
