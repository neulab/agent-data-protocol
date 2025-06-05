import json
import sys
from typing import Dict, List

from schema.action.action import Action
from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.image import BoundingBox, ImageAnnotation, ImageObservation
from schema.observation.observation import Observation
from schema.trajectory import Trajectory


def process_episode(episode_data: List[Dict]) -> Dict:
    """Process a list of data for a single episode into a standardized trajectory.

    Args:
        episode_data: List of data dictionaries for a single episode

    Returns:
        Standardized trajectory dictionary
    """
    if not episode_data:
        return None

    episode_id = episode_data[0]["episode_id"]
    content: list[Action | Observation] = []

    # Add the goal info as the first message
    content.append(MessageAction(content=episode_data[0]["goal_info"]))

    for data in episode_data:
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
                source="environment",
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
                    function="end",
                    kwargs={"succeeds": data["results/action_type"] == "task_complete"},
                )
            )
        else:
            raise ValueError(f"Unknown action type: {data['results/action_type']}")

    traj = Trajectory(id=episode_id, content=content)
    return traj.model_dump()


if __name__ == "__main__":
    # Process data line by line, but group by episode_id
    current_episode_id = None
    current_episode_data = []

    for line in sys.stdin:
        data = json.loads(line)

        # If we encounter a new episode, process the previous one
        if current_episode_id is not None and current_episode_id != data["episode_id"]:
            # Process and output the current episode
            standardized_trajectory = process_episode(current_episode_data)
            if standardized_trajectory:
                print(json.dumps(standardized_trajectory))
            # Start a new episode
            current_episode_data = [data]
        else:
            # Add to the current episode
            current_episode_data.append(data)

        current_episode_id = data["episode_id"]

    # Process the last episode
    if current_episode_data:
        standardized_trajectory = process_episode(current_episode_data)
        if standardized_trajectory:
            print(json.dumps(standardized_trajectory))
