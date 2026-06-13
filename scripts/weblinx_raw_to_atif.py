from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from schema.atif import Step
from scripts.legacy_atif import text_step, tool_step, trajectory, web_observation_step

INTENT_MAP = {
    "load": "goto",
    "click": "click",
    "textInput": "type",
    "paste": "type",
    "scroll": "scroll",
    "submit": "submit",
    "change": "select",
}

WEBLINX_DUMP = Path("datasets/weblinx/WebLINX-full")


def _safe_relative(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _xpath(args: dict[str, Any]) -> str:
    element = args.get("element") if isinstance(args.get("element"), dict) else {}
    attributes = element.get("attributes") if isinstance(element.get("attributes"), dict) else {}
    element_id = attributes.get("data-webtasks-id")
    return f"//*[@data-webtasks-id='{element_id}']" if element_id else str(element.get("xpath", ""))


def _web_observation(step: dict[str, Any], shortcode: str) -> Step:
    action = step.get("action") if isinstance(step.get("action"), dict) else {}
    args = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
    metadata = args.get("metadata") if isinstance(args.get("metadata"), dict) else {}
    state = step.get("state") if isinstance(step.get("state"), dict) else {}

    html = action.get("element_html") or ""
    axtree = ""
    page = state.get("page")
    if page:
        page_path = WEBLINX_DUMP / "demonstrations" / shortcode / "pages" / str(page)
        if page_path.exists():
            html = page_path.read_text()
            axtree = None

    image_path = None
    screenshot = state.get("screenshot")
    if screenshot:
        image_path = _safe_relative(
            WEBLINX_DUMP / "demonstrations" / shortcode / "screenshots" / str(screenshot)
        )

    viewport = None
    if metadata.get("viewportWidth") is not None and metadata.get("viewportHeight") is not None:
        viewport = [metadata["viewportWidth"], metadata["viewportHeight"]]

    return web_observation_step(
        html=html,
        axtree=axtree,
        url=metadata.get("url"),
        image_path=image_path,
        viewport_size=viewport,
    )


def _action_steps(step: dict[str, Any], shortcode: str) -> list[Step]:
    action = step.get("action") if isinstance(step.get("action"), dict) else {}
    intent = action.get("intent")
    if intent not in INTENT_MAP:
        return []

    args = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
    metadata = args.get("metadata") if isinstance(args.get("metadata"), dict) else {}
    function_name = INTENT_MAP[str(intent)]

    if intent == "load":
        return [tool_step(function_name, {"url": metadata.get("url", "")})]

    converted: list[Step] = [_web_observation(step, shortcode)]
    if intent == "scroll":
        converted.append(
            tool_step(function_name, {"dx": args.get("scrollX", 0), "dy": args.get("scrollY", 0)})
        )
        return converted

    if intent in {"click", "submit"}:
        converted.append(tool_step(function_name, {"xpath": _xpath(args)}))
        return converted

    value_key = {"textInput": "text", "paste": "pasted", "change": "value"}.get(str(intent))
    if value_key:
        converted.append(
            tool_step(function_name, {"xpath": _xpath(args), "value": args.get(value_key, "")})
        )
    return converted


def _convert_step(step: dict[str, Any], shortcode: str) -> list[Step]:
    if step.get("type") == "chat":
        speaker = step.get("speaker")
        if speaker == "instructor":
            return [text_step(str(step.get("utterance", "")), source="user")]
        if speaker == "navigator":
            return [text_step(str(step.get("utterance", "")), source="agent")]
        return []
    return _action_steps(step, shortcode)


def convert_record(raw_record: dict[str, Any], dataset_name: str):
    replay = raw_record.get("replay") if isinstance(raw_record.get("replay"), dict) else raw_record
    shortcode = str(raw_record.get("shortcode") or replay.get("shortcode") or "weblinx")
    form = raw_record.get("form") if isinstance(raw_record.get("form"), dict) else {}

    steps: list[Step] = []
    for raw_step in replay.get("data") or []:
        if isinstance(raw_step, dict):
            steps.extend(_convert_step(raw_step, shortcode))
    if not steps:
        steps = [text_step(json.dumps(raw_record, ensure_ascii=False))]

    return trajectory(
        dataset_name,
        shortcode,
        steps,
        raw=raw_record,
        details={
            "description": form.get("description"),
            "tasks": ", ".join(form.get("tasks") or []),
        },
    )


def main(script_file: str) -> None:
    dataset_name = script_file.rsplit("/", 2)[-2]
    for line in sys.stdin:
        if line.strip():
            print(convert_record(json.loads(line), dataset_name).model_dump_json(exclude_none=True))
