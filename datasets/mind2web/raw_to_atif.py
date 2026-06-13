# ruff: noqa: E402

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

from lxml import etree, html

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from schema.atif import Agent, ATIFObservation, ATIFTrajectory, ObservationResult, Step, ToolCall


def fix_iframes(html_str: str) -> str:
    document = html.fromstring(html_str)
    for iframe in document.xpath("//iframe"):
        if iframe.attrib.get("src") or iframe.attrib.get("srcdoc"):
            continue
        iframe_content = "".join(html.tostring(e, encoding="unicode") for e in iframe)
        iframe.attrib["srcdoc"] = iframe_content
        for child in iframe:
            iframe.remove(child)
    return etree.tostring(document, pretty_print=True, encoding="unicode", method="html")


def action_xpath(action: dict[str, Any]) -> str:
    label_xpath = f"//*[@data_pw_testid_buckeye='{action['action_uid']}']"
    tree = etree.HTML(action["raw_html"])
    elements = tree.xpath(label_xpath)
    backend_node_id = elements[0].get("backend_node_id") if elements else "not found"
    return f"//*[@backend_node_id='{backend_node_id}']"


def action_arguments(action: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    operation = action.get("operation") or {}
    function_name = str(operation.get("op") or "").lower()
    kwargs: dict[str, Any] = {"xpath": action_xpath(action)}
    if function_name in {"select", "type"} and operation.get("value") is not None:
        kwargs["value"] = operation["value"]
    return function_name, kwargs


def terminal_success(annotation_id: str) -> str:
    rng = random.Random(str(annotation_id))
    return rng.choice(
        [
            "Congratulations! You have successfully solved the task.",
            "Your solution has been verified as correct. ",
            "Well done on successfully completing the task!",
            "Your implementation satisfies the task requirements.",
            "Task completed successfully.",
        ]
    )


def finish_message(annotation_id: str) -> str:
    rng = random.Random(str(annotation_id))
    return rng.choice(
        [
            "I have successfully completed the task.",
            "I did it! The task is now complete.",
            "The objective has been achieved with no outstanding issues.",
            "I have fulfilled all the requirements of the task.",
            "I've wrapped up the task successfully.",
        ]
    )


def convert_record(record: dict[str, Any]) -> ATIFTrajectory:
    steps: list[Step] = [
        Step(
            step_id=1,
            source="user",
            message=f"Go to the website https://www.{record['website']}.com and {record['confirmed_task']}",
        ),
        Step(
            step_id=2,
            source="agent",
            message="",
            tool_calls=[
                ToolCall(
                    tool_call_id="call_000001",
                    function_name="goto",
                    arguments={"url": f"https://www.{record['website']}.com"},
                )
            ],
        ),
    ]
    next_step_id = 3
    next_call_id = 2
    for action in record.get("actions") or []:
        steps.append(
            Step(
                step_id=next_step_id,
                source="agent",
                message="",
                observation=ATIFObservation(
                    results=[
                        ObservationResult(
                            content="",
                            extra={
                                "web": {
                                    "html": fix_iframes(action["raw_html"]),
                                    "axtree": None,
                                    "url": f"https://www.{record['website']}.com",
                                }
                            },
                        )
                    ]
                ),
            )
        )
        next_step_id += 1
        function_name, arguments = action_arguments(action)
        steps.append(
            Step(
                step_id=next_step_id,
                source="agent",
                message="",
                tool_calls=[
                    ToolCall(
                        tool_call_id=f"call_{next_call_id:06d}",
                        function_name=function_name,
                        arguments=arguments,
                    )
                ],
            )
        )
        next_step_id += 1
        next_call_id += 1

    steps.append(
        Step(step_id=next_step_id, source="user", message=terminal_success(record["annotation_id"]))
    )
    next_step_id += 1
    steps.append(
        Step(
            step_id=next_step_id,
            source="agent",
            message="",
            tool_calls=[
                ToolCall(
                    tool_call_id=f"call_{next_call_id:06d}",
                    function_name="finish",
                    arguments={
                        "message": finish_message(record["annotation_id"]),
                        "task_completed": "true",
                    },
                )
            ],
        )
    )
    return ATIFTrajectory(
        trajectory_id=str(record["annotation_id"]),
        agent=Agent(name="mind2web", version="atif"),
        steps=steps,
        extra={"source_dataset": "mind2web"},
    )


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        print(convert_record(json.loads(line)).model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main()
