import json
import sys
from typing import Any

from schema.action.action import Action
from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.image import BoundingBox, ImageAnnotation, ImageObservation
from schema.observation.observation import Observation
from schema.trajectory import Trajectory

prev_id = None
content: list[Action | Observation] = []


def flush_episode(curr_data: dict[str, Any]):
    if content:
        traj = Trajectory(id=prev_id, content=list(content))
        print(traj.model_dump_json())
        content.clear()
    content.append(MessageAction(content=curr_data["goal_info"]))


for line in sys.stdin:
    data = json.loads(line)
    if prev_id != data["episode_id"]:
        flush_episode(data)
        prev_id = data["episode_id"]
    # Validating assumptions
    if data["goal_info"] != content[0].content:
        raise ValueError(
            "goal_info must be the same for all actions and observations in an episode"
            f" but got: {data['goal_info']} != {content[0].content}"
        )
    # Create the image observation
    annotations = [
        ImageAnnotation(
            text=text,
            element_type=ui_type,
            bounding_box=BoundingBox(
                x=pos[1],
                y=pos[0],
                width=pos[3],
                height=pos[2],
            ),
        )
        for text, ui_type, pos in zip(
            data["image/ui_annotations_text"],
            data["image/ui_annotations_ui_types"],
            data["image/ui_annotations_positions"],
        )
    ]
    content.append(
        ImageObservation(
            content=f"{data['image/encoded']}.png",
            annotations=annotations,
            source="os",
        )
    )
    # Create the action
    if data["results/action_type"] == "dual-point gesture":
        content.append(
            ApiAction(
                function="touch_and_lift",
                kwargs={
                    "x0": data["results/yx_touch"][1],
                    "y0": data["results/yx_touch"][0],
                    "x1": data["results/yx_lift"][1],
                    "y1": data["results/yx_lift"][0],
                },
            )
        )
    elif data["results/action_type"] == "type":
        content.append(ApiAction(function="type", kwargs={"text": data["results/type_action"]}))
    elif data["results/action_type"] in {"go_back", "go_home", "enter"}:
        content.append(
            ApiAction(function="press", kwargs={"key_name": data["results/action_type"]})
        )
    elif data["results/action_type"] in {"task_complete", "task_impossible"}:
        content.append(
            ApiAction(
                function="end", kwargs={"succeeds": data["results/action_type"] == "task_complete"}
            )
        )
    else:
        raise ValueError(f"Unknown action type: {data['results/action_type']}")


if content:
    traj = Trajectory(id=prev_id, content=content)
    print(traj.model_dump_json())
