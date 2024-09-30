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


def map_keypress(key: str) -> str:
    """
    Map keys for compatibility with playwright's keyboard.press
    https://playwright.dev/python/docs/api/class-keyboard#keyboard-press

    Args:
    ----
        key (str): The key to map.

    """
    key = key.strip("'")
    if len(key) == 1:
        return key
    if key.startswith("Key."):
        key = key[len("Key."):]
        if key.endswith("_r"):
            key = key[:-2]
        # capitalize the first letter
        key = key[0].upper() + key[1:]
        if key in ["Left", "Right", "Up", "Down"]:
            key = "Arrow" + key
        if key == "Cmd":
            key = "Meta"
    return key


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
                            "xpath": element["data"]["element_attributes"]["element"]["xpath"],
                        }
                        function_name = "click"
                    case "keystroke":
                        kwargs = {
                            "xpath": element["data"]["element_attributes"]["element"]["xpath"],
                            "value": ''.join(element["data"]["key"].strip("'").split("' '")), # "'h' 'e' 'l' 'l' 'o'" --> "hello"
                        }
                        function_name = "type"
                    case "keypress":
                        kwargs = {
                            "xpath": element["data"]["element_attributes"]["element"]["xpath"],
                            "value": map_keypress(element["data"]["key"])
                        }
                        function_name = "keyboard_press"
                    case "scroll":
                        kwargs = {
                            "dx": element["data"]["dx"],
                            "dy": element["data"]["dy"],
                        }
                        function_name = "scroll"
                    case _:
                        raise ValueError(f"Unknown action type: {function}")

                action = ApiAction(function=function_name, kwargs=kwargs)
                traj.content.append(action)
            except TypeError as e:
                continue
        else:
            raise ValueError(f"Unknown element type: {element['type']}")

    print(traj.model_dump_json())
