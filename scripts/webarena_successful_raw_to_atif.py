from __future__ import annotations

import json
import sys
from itertools import chain
from typing import Any

from schema.atif import Step
from scripts.legacy_atif import text_step, tool_step, trajectory, web_observation_step

ASCII_CHARSET = "".join(chr(x) for x in range(32, 128))
FREQ_UNICODE_CHARSET = "".join(chr(x) for x in range(129, 1000))
SPECIAL_KEYS = (
    "Enter",
    "Tab",
    "Control",
    "Shift",
    "Meta",
    "Backspace",
    "Delete",
    "Escape",
    "ArrowUp",
    "ArrowDown",
    "ArrowLeft",
    "ArrowRight",
    "PageDown",
    "PageUp",
    "Meta+a",
)
ID_TO_KEY = list(chain(SPECIAL_KEYS, ASCII_CHARSET, FREQ_UNICODE_CHARSET, ["\n"]))
SOURCE_BLACK_LIST = {"SteP"}
ROOT = "datasets/webarena_successful"


def _action_step(element: dict[str, Any]) -> Step:
    action = element["action"]
    function_name = action["action_name"]
    kwargs: dict[str, Any] = {}

    if function_name == "stop":
        kwargs["answer"] = action.get("answer", "")
    elif function_name == "type":
        kwargs["text"] = "".join(
            ID_TO_KEY[i]
            for i in action.get("text", [])
            if isinstance(i, int) and len(SPECIAL_KEYS) <= i < len(ID_TO_KEY)
        )
        kwargs["element_id"] = action.get("element_id")
    elif function_name in {"hover", "click"}:
        kwargs["element_id"] = action.get("element_id")
    elif function_name == "scroll":
        kwargs["dx"] = 0
        kwargs["dy"] = 100 if str(action.get("direction", "")).lower() == "down" else -100
    elif function_name in {"key_press", "press"}:
        kwargs["key_comb"] = action.get("key_comb")
        function_name = "press"
    elif function_name in {"new_tab", "goto", "goto_url"}:
        kwargs["url"] = action.get("url", "")
        function_name = "goto" if function_name == "goto_url" else function_name
    elif function_name in {"tab_focus", "page_focus"}:
        kwargs["page_number"] = action.get("page_number")
        function_name = "tab_focus"
    elif function_name in {"go_back", "page_close", "go_forward"}:
        function_name = "tab_close" if function_name == "page_close" else function_name
    else:
        raise ValueError(f"Unknown function: {function_name}")

    metadata = element.get("metadata") if isinstance(element.get("metadata"), dict) else {}
    return tool_step(function_name, kwargs, description=metadata.get("cot", ""))


def _web_observation(element: dict[str, Any]) -> Step:
    screenshot_path = str(element.get("screenshot_path", "")).replace(
        "demo_trajs/images/",
        f"{ROOT}/screenshots",
    )
    return web_observation_step(
        url=element.get("url"),
        html=element.get("axtree"),
        image_path=screenshot_path,
        viewport_size=[1280, 720],
    )


def convert_record(raw_record: dict[str, Any], dataset_name: str):
    if raw_record.get("source") in SOURCE_BLACK_LIST:
        return None

    steps: list[Step] = [text_step(str(raw_record.get("intent") or ""), source="user")]
    for element in raw_record.get("trajectory") or []:
        if not isinstance(element, dict):
            continue
        if "action" in element:
            steps.append(_action_step(element))
        elif "url" in element:
            steps.append(_web_observation(element))
        else:
            raise ValueError(f"Unknown WebArena element: {element}")

    return trajectory(dataset_name, str(raw_record.get("task_id")), steps, raw=raw_record)


def main(script_file: str) -> None:
    dataset_name = script_file.rsplit("/", 2)[-2]
    for line in sys.stdin:
        if not line.strip():
            continue
        converted = convert_record(json.loads(line), dataset_name)
        if converted is not None:
            print(converted.model_dump_json(exclude_none=True))
