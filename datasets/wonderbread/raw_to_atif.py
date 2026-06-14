from __future__ import annotations

import json
import os
import sys
from typing import Any

from schema.atif import Step
from scripts.legacy_atif import text_step, tool_step, trajectory, web_observation_step

ROOT = "datasets/wonderbread"


def map_keypress(key: str) -> str:
    key = key.strip("'")
    if len(key) == 1:
        return key
    if key.startswith("Key."):
        key = key[len("Key.") :]
        if key.endswith("_r"):
            key = key[:-2]
        key = key[0].upper() + key[1:]
        if key in {"Left", "Right", "Up", "Down"}:
            key = "Arrow" + key
        if key == "Cmd":
            key = "Meta"
    return key


def _task(raw_record: dict[str, Any]) -> str:
    webarena = raw_record.get("webarena")
    if isinstance(webarena, dict) and webarena.get("intent"):
        return str(webarena["intent"])
    return str(raw_record.get("task") or raw_record.get("sop") or "")


def _state_annotations(json_state: str | None) -> list[dict[str, Any]]:
    if not json_state:
        return []
    annotations = []
    for state in json.loads(json_state):
        annotations.append(
            {
                "text": state.get("text", ""),
                "element_type": state.get("tag"),
                "bbox": {
                    "x": state.get("x"),
                    "y": state.get("y"),
                    "width": state.get("width"),
                    "height": state.get("height"),
                },
            }
        )
    return annotations


def _state_axtree(json_state: str | None) -> str:
    annotations = _state_annotations(json_state)
    lines = []
    for index, annotation in enumerate(annotations, start=1):
        element_type = annotation.get("element_type") or "element"
        text = annotation.get("text") or ""
        lines.append(f"[{index}] {element_type} {text!r}")
    return "\n".join(lines)


def _state_step(element: dict[str, Any], task_stamp: str) -> Step:
    data = element["data"]
    screenshot = os.path.basename(str(data.get("path_to_screenshot", ""))).split(".")[0]
    screen_size = data.get("screen_size") if isinstance(data.get("screen_size"), dict) else {}
    json_state = data.get("json_state")
    return web_observation_step(
        url=data.get("url"),
        axtree=_state_axtree(json_state),
        image_path=f"{ROOT}/screenshots/{task_stamp}/{screenshot}.png",
        image_annotations=_state_annotations(json_state),
        viewport_size=[screen_size.get("width"), screen_size.get("height")]
        if screen_size
        else None,
    )


def _action_step(element: dict[str, Any]) -> Step | None:
    data = element["data"]
    function = data.get("type")
    attributes = data.get("element_attributes")
    element_attributes = attributes.get("element") if isinstance(attributes, dict) else {}
    xpath = element_attributes.get("xpath") if isinstance(element_attributes, dict) else None

    try:
        if function == "mouseup":
            return tool_step("click", {"xpath": xpath})
        if function == "keystroke":
            return tool_step(
                "type",
                {
                    "xpath": xpath,
                    "value": "".join(str(data.get("key", "")).strip("'").split("' '")),
                },
            )
        if function == "keypress":
            return tool_step("keyboard_press", {"xpath": xpath, "value": map_keypress(data["key"])})
        if function == "scroll":
            return tool_step("scroll", {"dx": data.get("dx"), "dy": data.get("dy")})
    except TypeError:
        return None
    raise ValueError(f"Unknown Wonderbread action type: {function}")


def convert_record(raw_record: dict[str, Any], dataset_name: str):
    task_stamp = str(raw_record.get("task_stamp") or "wonderbread")
    steps: list[Step] = [text_step(_task(raw_record), source="user")]
    for element in raw_record.get("trace") or []:
        if not isinstance(element, dict):
            continue
        if element.get("type") == "state":
            steps.append(_state_step(element, task_stamp))
        elif element.get("type") == "action":
            action = _action_step(element)
            if action is not None:
                steps.append(action)
        else:
            raise ValueError(f"Unknown Wonderbread element type: {element.get('type')}")
    return trajectory(dataset_name, task_stamp, steps, raw=raw_record)


def main(script_file: str) -> None:
    dataset_name = script_file.rsplit("/", 2)[-2]
    for line in sys.stdin:
        if line.strip():
            print(convert_record(json.loads(line), dataset_name).model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
