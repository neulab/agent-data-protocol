from typing import Any

from pydantic import BaseModel, ConfigDict


class Action(BaseModel):
    model_config = ConfigDict(extra="allow")

    action_type: str
    x: int | None = None
    y: int | None = None
    app_name: str | None = None
    direction: str | None = None


class SchemaRaw(BaseModel):
    model_config = ConfigDict(extra="allow")

    episode_id: int
    goal: str
    screenshots: list[str]
    accessibility_trees: list[Any]
    screenshot_widths: list[int]
    screenshot_heights: list[int]
    actions: list[Action]
    step_instructions: list[str]
