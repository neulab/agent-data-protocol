from typing import Any

from pydantic import Field, field_validator

from schema.action.action import Action


class ApiAction(Action):
    class_: str = Field("api_action", description="The class of the action")
    function: str
    kwargs: dict[str, Any]
    description: str | None = Field(
        None, description="The description/thought provided for the action"
    )

    @field_validator("class_")
    def validate_class(cls, v):
        if v != "api_action":
            raise ValueError(f"class_ must be 'api_action', got '{v}'")
        return v
