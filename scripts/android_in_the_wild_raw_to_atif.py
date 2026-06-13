from __future__ import annotations

import json
import sys
from pathlib import PureWindowsPath
from typing import Any

from schema.atif import Step
from scripts.legacy_atif import image_observation_step, text_step, tool_step, trajectory

ACTION_TYPES = {
    3: "type",
    4: "dual-point gesture",
    5: "go_back",
    6: "go_home",
    7: "enter",
    10: "task_complete",
    11: "task_impossible",
}


def image_observation_path(image_reference: str) -> str:
    filename = PureWindowsPath(image_reference).name
    if not filename.endswith(".png"):
        filename = f"{filename}.png"
    return f"images/{filename}"


def _action_type(record: dict[str, Any]) -> str:
    raw_type = record.get("results/action_type")
    if isinstance(raw_type, int):
        return ACTION_TYPES.get(raw_type, str(raw_type))
    return str(raw_type)


def _annotations(record: dict[str, Any]) -> list[dict[str, Any]]:
    annotations = []
    for text, element_type, pos in zip(
        record.get("image/ui_annotations_text") or [],
        record.get("image/ui_annotations_ui_types") or [],
        record.get("image/ui_annotations_positions") or [],
    ):
        if not isinstance(pos, list | tuple) or len(pos) < 4:
            continue
        annotations.append(
            {
                "text": text,
                "element_type": element_type,
                "bbox": {
                    "x": pos[1],
                    "y": pos[0],
                    "width": pos[3],
                    "height": pos[2],
                },
            }
        )
    return annotations


def _action_step(record: dict[str, Any]) -> Step:
    action_type = _action_type(record)
    if action_type == "dual-point gesture":
        yx_touch = record.get("results/yx_touch") or [0, 0]
        yx_lift = record.get("results/yx_lift") or yx_touch
        return tool_step(
            "touch_and_lift",
            {
                "x0": yx_touch[1],
                "y0": yx_touch[0],
                "x1": yx_lift[1],
                "y1": yx_lift[0],
            },
        )
    if action_type == "type":
        return tool_step("type", {"text": record.get("results/type_action", "")})
    if action_type in {"go_back", "go_home", "enter"}:
        return tool_step("press", {"key_name": action_type})
    if action_type in {"task_complete", "task_impossible"}:
        return tool_step("end", {"succeeds": action_type == "task_complete"})
    raise ValueError(f"Unknown action type: {action_type}")


def convert_episode(episode_data: list[dict[str, Any]], dataset_name: str):
    episode_id = str(episode_data[0]["episode_id"])
    goal = str(episode_data[0].get("goal_info") or "")
    steps: list[Step] = [text_step(goal, source="agent")]
    for record in episode_data:
        if record.get("goal_info") != episode_data[0].get("goal_info"):
            raise ValueError(f"goal_info changed inside episode {episode_id}")
        steps.append(
            image_observation_step(
                image_observation_path(str(record.get("image/encoded", "image"))),
                annotations=_annotations(record),
            )
        )
        steps.append(_action_step(record))
    return trajectory(dataset_name, episode_id, steps, raw={"episode": episode_data})


def main(script_file: str) -> None:
    dataset_name = script_file.rsplit("/", 2)[-2]
    current_episode_id = None
    current_episode: list[dict[str, Any]] = []
    for line in sys.stdin:
        if not line.strip():
            continue
        record = json.loads(line)
        episode_id = record.get("episode_id")
        if current_episode and current_episode_id != episode_id:
            print(convert_episode(current_episode, dataset_name).model_dump_json(exclude_none=True))
            current_episode = []
        current_episode.append(record)
        current_episode_id = episode_id
    if current_episode:
        print(convert_episode(current_episode, dataset_name).model_dump_json(exclude_none=True))
