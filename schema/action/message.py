from pydantic import Field

from schema.action.action import Action


class MessageAction(Action):
    class_: str = Field("message_action", description="The class of the action")
    content: str = Field(..., description="The message to share with the user")
    description: str | None = Field(
        None, description="The description/thought provided for the action"
    )
