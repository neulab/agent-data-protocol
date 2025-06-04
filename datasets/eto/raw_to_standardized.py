import json
import sys

from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

# Read the entire input as a JSON array
raw_data_list = []
for line in sys.stdin:
    if line.strip():
        raw_data_list.append(json.loads(line))
standardized_trajectories = []

for raw_data in raw_data_list:
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

    standardized_trajectories.append(traj.model_dump())

# Print the standardized data as JSONL (one JSON object per line)
for traj in standardized_trajectories:
    print(json.dumps(traj))
