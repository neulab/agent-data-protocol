from pydantic import Field, field_validator

from schema.action.action import Action


class MessageAction(Action):
    class_: str = Field("message_action", description="The class of the action")
    content: str = Field(..., description="The message to share with the user")
    description: str | None = Field(
        None, description="The description/thought provided for the action"
    )

    @field_validator("class_")
    def validate_class(cls, v):
        if v != "message_action":
            raise ValueError(f"class_ must be 'message_action', got '{v}'")
        return v
