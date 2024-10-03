import json
import sys

from typing import Any

from schema.action.action import Action
from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.image import BoundingBox, ImageObservation, ImageAnnotation
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory

from itertools import chain
from webarena_constants import SPECIAL_KEYS, ASCII_CHARSET, FREQ_UNICODE_CHARSET
_id2key = list(chain(SPECIAL_KEYS, ASCII_CHARSET, FREQ_UNICODE_CHARSET, ["\n"]))

root = "datasets/webarena_successful"
trajs: list[Trajectory] = []

# step does not have screenshot recording, skip to maintain the consistency of data format
SOURCE_BLACK_LIST = ['SteP']

for line in sys.stdin:
    raw_traj = json.loads(line)
    task = raw_traj["intent"]
    model = raw_traj["source"]
    if model in SOURCE_BLACK_LIST:
        continue
    traj = Trajectory(
        id=str(raw_traj["task_id"]),
        content=[TextObservation(content=task, source="user")],
    )
    for element in raw_traj["trajectory"]:
        if "action" in element:
            function = element["action"]["action_name"]
            kwargs = {}

            if function == "stop":
                kwargs["answer"] = element["action"]["answer"]
            elif function == "type":
                # TODO: ignoring the special keys for now, fix this
                kwargs["text"] = ''.join([_id2key[i] for i in element["action"]["text"] if i < len(_id2key) and i >= len(SPECIAL_KEYS)])
                kwargs["element_id"] = element["action"]["element_id"]
            elif function in ["hover", "click"]:
                kwargs["element_id"] = element["action"]["element_id"]
            elif function == "scroll":
                kwargs["dx"] = 0
                kwargs["dy"] = 100 if element["action"]["direction"].lower() == "down" else -100
            elif function in ["key_press", "press"]:
                kwargs["key_comb"] = element["action"]["key_comb"]
                function = "press"
            elif function in ["new_tab", "goto", "goto_url"]:
                kwargs["url"] = element["action"]["url"]
                function = "goto" if function == "goto_url" else function
            elif function in ['tab_focus', 'page_focus']:
                kwargs["page_number"] = element["action"]["page_number"]
                function = "tab_focus"
            elif function in ['go_back', 'page_close', 'go_forward']:
                kwargs = {}
                function = "tab_close" if function == "page_close" else function
            else:
                raise ValueError(f"Unknown function: {function}")

            action = ApiAction(
                function=function,
                kwargs=kwargs,
                description=element["metadata"].get("cot", ""),
            )
            traj.content.append(action)
        elif "url" in element:
            url = element["url"]
            html = element["axtree"]
            screenshot_path = element["screenshot_path"].replace(
                "demo_trajs/images/", f"{root}/screenshots"
            )
            img_obs = ImageObservation(
                content=screenshot_path,
                annotations=[],
                source="browser",
            )
            web_obs = WebObservation(
                url=url,
                viewport_size=(1280, 720),
                html=html,
                image_observation=img_obs,
            )
            traj.content.append(web_obs)
        else:
            raise ValueError(f"Unknown element type: {element}")
    trajs.append(traj)

for traj in trajs:
    print(traj.model_dump_json())
