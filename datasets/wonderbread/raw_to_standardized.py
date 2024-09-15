import collections
import json
import os
import sys
from typing import Any

from schema.action.action import Action
from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.observation.image import BoundingBox, ImageObservation, ImageAnnotation
from schema.trajectory import Trajectory

root = "datasets/wonderbread"

for line in sys.stdin:
    raw_traj = json.loads(line)
    task = raw_traj["task"]
    task_stamp = raw_traj["task_stamp"]
    sop = raw_traj["sop"]

    traj: Trajectory = Trajectory(
        id=task_stamp,
        # task=task,
        content=[
            TextObservation(content=task, source="user")
        ],  # first message is the task
    )
    for element in raw_traj["trace"]:
        if element["type"] == "state":
            html = element["data"]["html"]
            url = element["data"]["url"]

            json_state = json.loads(element["data"]["json_state"])
            annotations = []
            for state in json_state:
                bbox = BoundingBox(
                    x=state["x"],
                    y=state["y"],
                    width=state["width"],
                    height=state["height"],
                )
                element_type = state["tag"]
                text = state.get("text", "")
                xpath = state.get("xpath", None)
                annotation = ImageAnnotation(
                    text=text, element_type=element_type, bounding_box=bbox, xpath=xpath
                )
                annotations.append(annotation)
            image_observation = ImageObservation(
                content=f"{root}/screenshots/{task_stamp}/{os.path.basename(element['data']['path_to_screenshot']).split('.')[0]}.png",
                annotations=annotations,
                source="browser",
            )

            web_obs = WebObservation(
                url=url,
                viewport_size=(
                    element["data"]["screen_size"]["width"],
                    element["data"]["screen_size"]["height"],
                ),
                html=html,
                image_observation=image_observation,
            )
            traj.content.append(web_obs)

        elif element["type"] == "action":
            function = element["data"]["type"]
            try:
                match function:
                    case "mouseup":
                        kwargs = {
                            "element": element["data"]["element_attributes"]["element"],
                            "misc": {
                                "is_right_click": element["data"]["is_right_click"],
                                "pressed": element["data"]["pressed"],
                            },
                        }
                    case "keystroke":
                        kwargs = {
                            "element": element["data"]["element_attributes"]["element"],
                            "str": element["data"]["key"],
                        }
                    case "keypress":
                        kwargs = {"key": element["data"]["key"]}
                    case "scroll":
                        kwargs = {
                            "x": element["data"]["dx"],
                            "y": element["data"]["dy"],
                        }
                    case _:
                        raise ValueError(f"Unknown action type: {function}")

                action = ApiAction(function=function, kwargs=kwargs)
                traj.content.append(action)
            except TypeError as e:
                continue
        else:
            raise ValueError(f"Unknown element type: {element['type']}")

    print(traj.model_dump_json())
