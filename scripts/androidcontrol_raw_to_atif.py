from __future__ import annotations

import json
import sys
from typing import Any

from schema.atif import Step
from scripts.legacy_atif import image_observation_step, text_step, tool_step, trajectory


def _action_step(action: dict[str, Any], description: str | None) -> Step | None:
    action_type = action.get("action_type")
    if action_type in {"click", "long_press"}:
        return tool_step(
            "click",
            {"x": action.get("x"), "y": action.get("y")},
            description=description,
        )
    if action_type == "scroll":
        return tool_step("scroll", {"direction": action.get("direction")}, description=description)
    if action_type == "input_text":
        return tool_step("input_text", {"text": action.get("text", "")}, description=description)
    if action_type == "navigate_home":
        return tool_step("navigate_home", {}, description=description)
    if action_type == "navigate_back":
        return tool_step("back", {}, description=description)
    if action_type == "open_app":
        return tool_step("open_app", {"app_name": action.get("app_name", "")}, description=description)
    if action_type == "wait":
        return tool_step("wait", {}, description=description)
    return None


def convert_record(record: dict[str, Any], dataset_name: str):
    episode_id = str(record.get("episode_id", "androidcontrol"))
    screenshots = record.get("screenshots") or []
    actions = record.get("actions") or []
    instructions = record.get("step_instructions") or []
    steps: list[Step] = [text_step(str(record.get("goal") or ""), source="agent")]

    for index, screenshot in enumerate(screenshots):
        steps.append(image_observation_step(str(screenshot), source="user"))
        if index >= len(actions):
            continue
        action_step = _action_step(
            actions[index],
            str(instructions[index]) if index < len(instructions) else None,
        )
        if action_step is not None:
            steps.append(action_step)

    return trajectory(dataset_name, episode_id, steps, raw=record)


def main(script_file: str) -> None:
    dataset_name = script_file.rsplit("/", 2)[-2]
    for line in sys.stdin:
        if line.strip():
            print(convert_record(json.loads(line), dataset_name).model_dump_json(exclude_none=True))
