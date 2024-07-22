import json
import sys
from typing import Any

from schema.action.action import Action
from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.image import BoundingBox, ImageObservation, ImageAnnotation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory
from typing import Any, List, Union

prev_id = None
content: list[Action | Observation] = []


def convert_to_trajectory(data: dict[str, Any]) -> Trajectory:
    content: List[Union[ApiAction, MessageAction, TextObservation, ImageObservation]] = []
    # print(data)
    # Add the goal as a MessageAction
    content.append(MessageAction(content=data["goal"]))
    #
    # # Add the image observations
    for i, (screenshot, tree) in enumerate(zip(data["screenshots"], data["accessibility_trees"])):
        annotations = []
        for element in tree:

            element_text = ""
            if element["text"]:
                element_text = element["text"]
                # Special handling of single character labels.
                if (
                        len(element_text) == 1
                        and element["content_description"]
                        and len(element["content_description"]) > 1
                ):
                    element_text = element["content_description"]
            elif element["content_description"]:
                element_text = element["content_description"]
            elif element["hint_text"]:
                element_text = element["hint_text"]
            elif element["tooltip"]:
                element_text = element["tooltip"]

            elif element["class_name"] and element["class_name"].endswith('Switch'):
                element_text = 'Switch:' + ('on' if element["is_checked"] else 'off')
            elif element["resource_id"]:
                element_text = element.resource_id.split('/')[-1]
            elif element["class_name"] and element["class_name"].endswith(
                    'EditText'
            ):
                element_text = element["edit text"]
            else:
                element_text = ""
            image_annotation = ImageAnnotation(
                # text=element["text"],
                # element_type=element["class_name"],

                text=element_text if element_text else "",
                element_type=element["class_name"] if element["class_name"] else "",
                bounding_box=BoundingBox(
                    x=element["bbox_pixels"]["x_min"],
                    y=element["bbox_pixels"]["y_min"],
                    width=element["bbox_pixels"]["width"],
                    height=element["bbox_pixels"]["height"]
                )
            )
            annotations.append(image_annotation)
        content.append(
            ImageObservation(
                content=screenshot,
                annotations=annotations,
                source="user"
            )
        )
        if i!=len(data["screenshots"])-1:
            action=data["actions"][i]
            step_inst=data["step_instructions"][i]
            if action["action_type"] == "click":
                # print("click")
                content.append(
                    ApiAction(
                        function="click",
                        kwargs={"x": action["x"], "y": action["y"]},
                        description=step_inst
                    )
                )
            elif action["action_type"] == "long_press":
                content.append(
                    ApiAction(
                        function="click",
                        kwargs={"x": action["x"], "y": action["y"]},
                        description=step_inst
                    )
                )
            elif action["action_type"] == "scroll":
                content.append(
                    ApiAction(
                        function="scroll",
                        kwargs={"direction": action["direction"]},
                        description=step_inst
                    )
                )
            elif action["action_type"] == "input_text":
                content.append(
                    ApiAction(
                        function="input_text",
                        kwargs={"text": action["text"]},
                        description=step_inst
                    )
                )
            elif action["action_type"] == "navigate_home":
                content.append(
                    ApiAction(
                        function="navigate_home",
                        kwargs={},
                        description=step_inst
                    )
                )
            elif action["action_type"] == "navigate_back":
                content.append(
                    ApiAction(
                        function="back",
                        kwargs={},
                        description=step_inst
                    )
                )
            elif action["action_type"] == "open_app":
                content.append(
                    ApiAction(
                        function="open_app",
                        kwargs={"app_name": action["app_name"]},
                        description=step_inst
                    )
                )
            elif action["action_type"] == "wait":
                content.append(
                    ApiAction(
                        function="wait",
                        kwargs={},
                        description=step_inst
                    )
                )
    # print(content)
    return Trajectory(id=str(data["episode_id"]), content=content)


# Load the JSON data
file_path = "sample_raw.json"
with open(file_path, "r") as f:
    raw_data = json.load(f)

for i in raw_data:
    # Convert and print the results
    trajectory = convert_to_trajectory(i)
    # print(trajectory.json(indent=2))

    output_file = "output_trajectory.json"
    with open(output_file, "w") as f:
        f.write(trajectory.model_dump_json(indent=2))
    print(trajectory.model_dump_json())

# print(f"Trajectory saved to {output_file}")
