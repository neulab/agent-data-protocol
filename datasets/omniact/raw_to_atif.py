from __future__ import annotations

import json
import re
import sys
from typing import Any

from schema.atif import Step
from scripts.legacy_atif import image_observation_step, text_step, tool_step, trajectory


def _annotations(record: dict[str, Any]) -> list[dict[str, Any]]:
    annotations = []
    ocr = record.get("ocr") if isinstance(record.get("ocr"), dict) else {}
    boxes = record.get("box") if isinstance(record.get("box"), dict) else {}
    texts = ocr.get("text") or []
    top_left = boxes.get("top_left") or []
    bottom_right = boxes.get("bottom_right") or []
    for text, start, end in zip(texts, top_left, bottom_right):
        if not isinstance(start, list | tuple) or not isinstance(end, list | tuple):
            continue
        if len(start) < 2 or len(end) < 2:
            continue
        annotations.append(
            {
                "text": text,
                "element_type": "text",
                "bbox": {
                    "x": start[0],
                    "y": start[1],
                    "width": end[0] - start[0],
                    "height": end[1] - start[1],
                },
            }
        )
    return annotations


def _split_task_and_script(task: str) -> tuple[str, str | None]:
    match = re.search(r"\bOutput Script:\s*(.*)$", task, flags=re.DOTALL)
    if not match:
        return task.strip(), None
    return task[: match.start()].strip(), match.group(1).strip()


def convert_record(record: dict[str, Any], dataset_name: str):
    task, script = _split_task_and_script(str(record.get("task") or ""))
    steps: list[Step] = [
        text_step(task, source="user"),
        image_observation_step(
            str(record.get("image") or "image"), annotations=_annotations(record)
        ),
    ]
    if script:
        steps.append(tool_step("execute_ipython_cell", {"code": script}, message="Output Script"))
    return trajectory(dataset_name, str(record.get("id")), steps, raw=record)


def main(script_file: str) -> None:
    dataset_name = script_file.rsplit("/", 2)[-2]
    for line in sys.stdin:
        if line.strip():
            print(convert_record(json.loads(line), dataset_name).model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
