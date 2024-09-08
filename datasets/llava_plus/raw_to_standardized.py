import json
import sys
import os

from schema.action.action import Action
from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.observation.image import ImageObservation
from schema.trajectory import Trajectory


def convert_step(step: dict[str, str], metadata) -> list[Action | Observation]:
    if step["from"] == "human":
        if step["value"].startswith("<image>\n"):
            return [
                ImageObservation(
                    content=os.path.join("images/", metadata["data_source"], metadata["image"]),
                    annotations=None,
                    source="system"
                ),
                TextObservation(content=step["value"][len("<image>\n"):], source="user")
            ] 
        else:
            return [
                TextObservation(content=step["value"], source="user")
            ]
    elif step["from"] == "gpt":
        if len(step["actions"]) > 0:
            content = [
                MessageAction(
                    content=step["value"],
                    description=step["thoughts"],
                )
            ]

            for act in step["actions"]:
                content.extend([
                    ApiAction(
                        function=act["API_name"],
                        kwargs=act["API_params"],
                        description=None,
                    )
                ])

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


for line in sys.stdin:
    raw_data = json.loads(line)

    metadata = dict([(k, v) for k, v in raw_data.items() if k != "conversations"])

    content = []
    for step in raw_data["conversations"]:
        content.extend(convert_step(step, metadata))

    # Standardize the data
    standardize_data = Trajectory(
        id=str(raw_data["unique_id"]),
        content=content
    )

    # Print the standardized data
    print(standardize_data.model_dump_json())
