import json
import sys

from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

# Process each line of input individually
for line in sys.stdin:
    raw_data = json.loads(line)
    content = []

    # Process the conversations
    for message in raw_data["conversations"]:
        if message["from"] == "human":
            content.append(TextObservation(content=message["value"], source="user"))
        elif message["from"] == "gpt":
            content.append(MessageAction(content=message["value"], description=""))

    # Handle finish actions for message actions
    if isinstance(content[-1], MessageAction) and "<finish>" not in content[-1].content:
        content[-1].content = f"<finish> {content[-1].content} </finish>"

    # Create trajectory
    traj = Trajectory(id=raw_data["id"], content=content)

    # Print the standardized data as JSON
    print(json.dumps(traj.model_dump()))
