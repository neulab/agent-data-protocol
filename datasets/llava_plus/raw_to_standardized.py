import json
import os
import sys

from schema.action.action import Action
from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.image import ImageObservation
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


def convert_step(step: dict[str, str], metadata) -> list[Action | Observation]:
    if step["from"] == "human":
        if step["value"].startswith("<image>\n"):
            return [
                ImageObservation(
                    content=os.path.join("images/", metadata["data_source"], metadata["image"]),
                    annotations=None,
                    source="environment",
                ),
                TextObservation(content=step["value"][len("<image>\n") :], source="user"),
            ]
        else:
            return [TextObservation(content=step["value"], source="user")]
    elif step["from"] == "gpt":
        if len(step["actions"]) > 0:
            content = [
                MessageAction(
                    content=step["value"],
                    description=step["thoughts"],
                )
            ]

            for act in step["actions"]:
                content.extend(
                    [
                        ApiAction(
                            function=act["API_name"],
                            kwargs=act["API_params"],
                            description=None,
                        )
                    ]
                )

            return content
        else:
            return [
                MessageAction(
                    content=step["value"],
                    description=step["thoughts"],
                )
            ]
    else:
        raise Exception("Invalid role.")


# Read the entire input as a JSON array
raw_data_list = []
for line in sys.stdin:
    if line.strip():
        raw_data_list.append(json.loads(line))
standardized_trajectories = []

for raw_data in raw_data_list:
    metadata = dict([(k, v) for k, v in raw_data.items() if k != "conversations"])

    content = []
    for step in raw_data["conversations"]:
        content.extend(convert_step(step, metadata))

    # Standardize the data
    standardize_data = Trajectory(id=str(raw_data["unique_id"]), content=content)
    standardized_trajectories.append(standardize_data.model_dump())

# Print the standardized data as JSONL (one JSON object per line)
for traj in standardized_trajectories:
    print(json.dumps(traj))
