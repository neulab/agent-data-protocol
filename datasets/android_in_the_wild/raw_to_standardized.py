import json
import sys
from typing import Dict, List

from schema.action.action import Action
from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.image import BoundingBox, ImageAnnotation, ImageObservation
from schema.observation.observation import Observation
from schema.trajectory import Trajectory

# Constants from Android in the Wild action matching code
_SWIPE_DISTANCE_THRESHOLD = 0.04


def _is_tap(touch_yx: List[float], lift_yx: List[float]) -> bool:
    """Check if a dual-point gesture is a tap (touch and lift are close together).

    Args:
        touch_yx: The (y, x) coordinates where the touch started
        lift_yx: The (y, x) coordinates where the touch lifted

    Returns:
        True if the action is a tap, False if it's a swipe
    """
    distance = ((touch_yx[0] - lift_yx[0]) ** 2 + (touch_yx[1] - lift_yx[1]) ** 2) ** 0.5
    return distance <= _SWIPE_DISTANCE_THRESHOLD


def _point_in_bbox(point_yx: List[float], bbox: List[float]) -> bool:
    """Check if a point falls within a bounding box.

    Args:
        point_yx: The (y, x) coordinates of the point (normalized 0-1)
        bbox: The bounding box as [y, x, height, width] (normalized 0-1)

    Returns:
        True if the point is within the bounding box
    """
    y, x = point_yx
    bbox_y, bbox_x, bbox_height, bbox_width = bbox

    # Check if point is within the box boundaries
    return bbox_y <= y <= bbox_y + bbox_height and bbox_x <= x <= bbox_x + bbox_width


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

    # Pass 1: Analyze actions to determine clickability and editability
    # Structure: {step_id: {annotation_idx: {"clickable": bool, "editable": bool}}}
    annotation_properties = {}

    for idx, data in enumerate(episode_data):
        step_id = data["step_id"]
        annotation_properties[step_id] = {}

        # Initialize properties for each annotation
        num_annotations = len(data["image/ui_annotations_positions"])
        for ann_idx in range(num_annotations):
            annotation_properties[step_id][ann_idx] = {"clickable": False, "editable": False}

        # Mark ICON_* elements as clickable
        for ann_idx, ui_type in enumerate(data["image/ui_annotations_ui_types"]):
            if ui_type.startswith("ICON_"):
                annotation_properties[step_id][ann_idx]["clickable"] = True

        # Check if current action is a tap and mark the tapped element as clickable
        if data["results/action_type"] == "dual-point gesture":
            touch_yx = data["results/yx_touch"]
            lift_yx = data["results/yx_lift"]

            if _is_tap(touch_yx, lift_yx):
                # Find which annotation contains the tap point
                for ann_idx, bbox in enumerate(data["image/ui_annotations_positions"]):
                    if _point_in_bbox(touch_yx, bbox):
                        annotation_properties[step_id][ann_idx]["clickable"] = True

                        # Check if next action is type, then mark as editable
                        if idx + 1 < len(episode_data):
                            next_data = episode_data[idx + 1]
                            if next_data["results/action_type"] == "type":
                                annotation_properties[step_id][ann_idx]["editable"] = True
                        break  # Only mark the first matching bounding box

    # Pass 2: Create trajectory content with enhanced annotations
    for data in episode_data:
        # Validating assumptions
        if data["goal_info"] != content[0].content:
            raise ValueError(
                "goal_info must be the same for all actions and observations in an episode"
                f" but got: {data['goal_info']} != {content[0].content}"
            )
        # Create the image observation
        step_id = data["step_id"]
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
                clickable=annotation_properties[step_id][ann_idx]["clickable"],
                editable=annotation_properties[step_id][ann_idx]["editable"],
            )
            for ann_idx, (text, ui_type, pos) in enumerate(
                zip(
                    data["image/ui_annotations_text"],
                    data["image/ui_annotations_ui_types"],
                    data["image/ui_annotations_positions"],
                )
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

    # Read all input first to detect format
    input_text = sys.stdin.read()

    # Try to parse as a JSON array first (for sample files)
    try:
        all_data = json.loads(input_text)
        # If it's a single object, wrap it in a list
        if isinstance(all_data, dict):
            all_data = [all_data]
    except json.JSONDecodeError:
        # Fall back to newline-delimited JSON
        all_data = [json.loads(line) for line in input_text.strip().split("\n") if line.strip()]

    for data in all_data:
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
