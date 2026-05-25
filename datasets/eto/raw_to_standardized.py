import json
import re
import sys

from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.tool_call_links import create_trajectory_with_tool_call_links

ACTION_RE = re.compile(
    r"(?P<thought>.*?)\bAction:\s*(?P<function>search|click)\[(?P<value>.*?)\]\s*$",
    re.DOTALL | re.IGNORECASE,
)


def parse_webshop_action(content: str) -> ApiAction | None:
    match = ACTION_RE.match(content.strip())
    if match is None:
        return None

    function = match.group("function").lower()
    value = match.group("value").strip()
    thought = match.group("thought").strip()
    if thought.startswith("Thought:"):
        thought = thought[len("Thought:") :].strip()

    if function == "search":
        return ApiAction(function="search", kwargs={"keywords": value}, description=thought)
    return ApiAction(function="click", kwargs={"bid": json.dumps(value)}, description=thought)


# Process each line of input individually
for line in sys.stdin:
    raw_data = json.loads(line)
    content = []

    # Process the conversations
    for message in raw_data["conversations"]:
        if message["from"] == "human":
            content.append(TextObservation(content=message["value"], source="user"))
        elif message["from"] == "gpt":
            action = parse_webshop_action(message["value"])
            if action is not None:
                content.append(action)
            else:
                content.append(MessageAction(content=message["value"], description=""))

    # WebShop final click actions terminate the successful interaction and do not
    # have a following raw environment observation.
    if isinstance(content[-1], ApiAction):
        content.append(
            TextObservation(content="Task completed successfully.", source="environment")
        )
        content.append(
            MessageAction(
                content="<finish> I have successfully completed the task. </finish>",
                description="",
            )
        )

    # Handle finish actions for message actions.
    if isinstance(content[-1], MessageAction) and "<finish>" not in content[-1].content:
        content[-1].content = f"<finish> {content[-1].content} </finish>"

    # Create trajectory
    traj = create_trajectory_with_tool_call_links(id=raw_data["id"], content=content)

    # Print the standardized data as JSON
    print(json.dumps(traj.model_dump()))
