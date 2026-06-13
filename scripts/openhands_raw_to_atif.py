from __future__ import annotations

import ast
import json
import sys
from typing import Any

from schema.atif import Step
from scripts.legacy_atif import text_step, tool_step, trajectory, web_observation_step
from scripts.raw_to_atif_common import add_observation


def _extras(step: dict[str, Any]) -> dict[str, Any]:
    extras = step.get("extras")
    if isinstance(extras, dict):
        return extras
    if isinstance(extras, str) and extras:
        try:
            parsed = json.loads(extras)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _parse_browser_action(action: str) -> tuple[str | None, dict[str, Any]]:
    try:
        expression = ast.parse(action, mode="eval")
    except SyntaxError:
        return None, {}
    if not isinstance(expression.body, ast.Call) or not isinstance(expression.body.func, ast.Name):
        return None, {}
    call = expression.body
    args = [ast.literal_eval(arg) for arg in call.args]
    kwargs = {
        keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords if keyword.arg
    }
    function_name = call.func.id
    positional_names = {
        "goto": ["url"],
        "click": ["bid"],
        "dblclick": ["bid"],
        "hover": ["bid"],
        "press": ["bid", "key_comb"],
        "fill": ["bid", "value"],
        "select_option": ["bid", "options"],
        "send_msg_to_user": ["msg"],
        "scroll": ["delta_x", "delta_y"],
    }.get(function_name, [])
    for name, value in zip(positional_names, args):
        kwargs.setdefault(name, value)
    return function_name, kwargs


def _append_observation(steps: list[Step], step: dict[str, Any], extras: dict[str, Any]) -> None:
    observation = step.get("observation")
    content = step.get("content") or step.get("message") or ""
    if observation == "browse":
        add_observation(
            steps,
            web_observation_step(
                url=extras.get("url"),
                html=step.get("content"),
                axtree=extras.get("axtree_object"),
                image_path=extras.get("screenshot"),
            )
            .observation.results[0]
            .content,
        )
        return
    if observation in {
        "run",
        "run_ipython",
        "edit",
        "write",
        "read",
        "rag_search",
        "crawl",
        "delegate",
        "task_plan",
        "error",
        "user_rejected",
    }:
        add_observation(steps, str(content))


def convert_record(record: dict[str, Any], dataset_name: str):
    steps: list[Step] = []
    for raw_step in record.get("trajectory") or []:
        if not isinstance(raw_step, dict):
            continue
        action = raw_step.get("action")
        observation = raw_step.get("observation")
        extras = _extras(raw_step)

        if not action and observation:
            _append_observation(steps, raw_step, extras)
            continue
        if not action:
            continue

        if action == "message":
            content = extras.get("content") or raw_step.get("content") or ""
            if not content:
                continue
            source = "user" if raw_step.get("source") == "user" else "agent"
            steps.append(text_step(str(content), source=source))
        elif action == "run_ipython":
            steps.append(
                tool_step(
                    "execute_ipython_cell",
                    {"code": extras.get("code", "")},
                    message=str(extras.get("thought") or ""),
                )
            )
        elif action == "run":
            steps.append(
                tool_step(
                    "execute_bash",
                    {"command": extras.get("command", "")},
                    message=str(extras.get("thought") or ""),
                )
            )
        elif action == "browse_interactive":
            thought = str(extras.get("thought") or "")
            for browser_action in str(extras.get("browser_actions") or "").splitlines():
                browser_action = browser_action.strip()
                if not browser_action:
                    continue
                function_name, arguments = _parse_browser_action(browser_action)
                if function_name:
                    steps.append(tool_step(function_name, arguments, message=thought))
                    thought = ""
        elif action == "finish":
            output = (
                extras.get("outputs", {}).get("content")
                if isinstance(extras.get("outputs"), dict)
                else None
            )
            steps.append(
                tool_step(
                    "finish",
                    {
                        "message": output or raw_step.get("message") or "",
                        "task_completed": "true",
                    },
                    message=str(extras.get("thought") or raw_step.get("message") or ""),
                )
            )

    if not steps:
        steps = [text_step(json.dumps(record, ensure_ascii=False))]
    return trajectory(dataset_name, str(record.get("id")), steps, raw=record)


def main(script_file: str) -> None:
    dataset_name = script_file.rsplit("/", 2)[-2]
    for line in sys.stdin:
        if line.strip():
            print(convert_record(json.loads(line), dataset_name).model_dump_json(exclude_none=True))
