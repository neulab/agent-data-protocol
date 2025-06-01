from typing import Any

from pydantic import Field

from schema.action.action import Action


class ApiAction(Action):
    function: str
    kwargs: dict[str, Any]
    description: str | None = Field(
        None, description="The description/thought provided for the action"
    )
