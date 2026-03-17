import json
import sys
from typing import Any

sys.path.append(".")

from typing import List, Union

from schema.action.action import Action
from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.image import BoundingBox, ImageAnnotation, ImageObservation
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


def convert_to_trajectory(data: dict[str, Any]) -> Trajectory:
    content: List[Union[ApiAction, MessageAction, TextObservation, ImageObservation]] = []

    # Add the goal as the first user observation (maps to "human" in SFT)
    content.append(TextObservation(content=data["goal"], source="user"))
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

            elif element["class_name"] and element["class_name"].endswith("Switch"):
                element_text = "Switch:" + ("on" if element["is_checked"] else "off")
            elif element["resource_id"]:
                element_text = element["resource_id"].split("/")[-1]
            elif element["class_name"] and element["class_name"].endswith("EditText"):
                element_text = element["edit text"]
            else:
                element_text = ""

            # Build natural language content description from non-visual metadata
            desc_parts = []
            if element["resource_id"]:
                desc_parts.append(f"Resource ID: {element['resource_id']}")
            # Include hint_text if it wasn't already used in element_text
            if element["hint_text"] and element["hint_text"] != element_text:
                desc_parts.append(f"Hint: {element['hint_text']}")
            # Include tooltip if it wasn't already used in element_text
            if element["tooltip"] and element["tooltip"] != element_text:
                desc_parts.append(f"Tooltip: {element['tooltip']}")

            content_desc = ", ".join(desc_parts) if desc_parts else None

            image_annotation = ImageAnnotation(
                # text=element["text"],
                # element_type=element["class_name"],
                text=element_text if element_text else "",
                element_type=element["class_name"] if element["class_name"] else "",
                bounding_box=BoundingBox(
                    x=element["bbox_pixels"]["x_min"],
                    y=element["bbox_pixels"]["y_min"],
                    width=element["bbox_pixels"]["width"],
                    height=element["bbox_pixels"]["height"],
                ),
                content_description=content_desc,
                clickable=element["is_clickable"],
                editable=element["is_editable"],
            )
            annotations.append(image_annotation)
        content.append(ImageObservation(content=f"datasets/androidcontrol/screenshots/{screenshot}", annotations=annotations, source="user"))
        if i != len(data["screenshots"]) - 1:
            action = data["actions"][i]
            step_inst = data["step_instructions"][i]
            if action["action_type"] == "click":
                content.append(
                    ApiAction(
                        function="click",
                        kwargs={"x": action["x"], "y": action["y"]},
                        description=step_inst,
                    )
                )
            elif action["action_type"] == "long_press":
                content.append(
                    ApiAction(
                        function="click",
                        kwargs={"x": action["x"], "y": action["y"]},
                        description=step_inst,
                    )
                )
            elif action["action_type"] == "scroll":
                content.append(
                    ApiAction(
                        function="scroll",
                        kwargs={"direction": action["direction"]},
                        description=step_inst,
                    )
                )
            elif action["action_type"] == "input_text":
                content.append(
                    ApiAction(
                        function="input_text",
                        kwargs={"text": action["text"]},
                        description=step_inst,
                    )
                )
            elif action["action_type"] == "navigate_home":
                content.append(
                    ApiAction(function="navigate_home", kwargs={}, description=step_inst)
                )
            elif action["action_type"] == "navigate_back":
                content.append(ApiAction(function="back", kwargs={}, description=step_inst))
            elif action["action_type"] == "open_app":
                content.append(
                    ApiAction(
                        function="open_app",
                        kwargs={"app_name": action["app_name"]},
                        description=step_inst,
                    )
                )
            elif action["action_type"] == "wait":
                content.append(ApiAction(function="wait", kwargs={}, description=step_inst))
    return Trajectory(id=str(data["episode_id"]), content=content)


record_count = 0
error_count = 0
for line in sys.stdin:
    try:
        raw_data = json.loads(line)
        trajectory = convert_to_trajectory(raw_data)
        print(trajectory.model_dump_json())
        record_count += 1
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        error_count += 1
        print(f"Warning: Skipping record: {e}", file=sys.stderr)

print(f"Processed {record_count} episodes ({error_count} errors)", file=sys.stderr)
