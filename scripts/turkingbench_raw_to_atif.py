from __future__ import annotations

import json
import sys
import urllib.parse
from typing import Any

from lxml import etree

from schema.atif import Step
from scripts.legacy_atif import text_step, tool_step, trajectory, web_observation_step

INPUT_ELEMENTS = [
    "input",
    "textarea",
    "select",
    "crowd-checkbox",
    "crowd-slider",
    "crowd-input",
]
ALL_INPUT_ELEMENTS_XPATH = " | ".join(f"//{element}" for element in INPUT_ELEMENTS)
ACTIONS = {
    "radio": "click",
    "checkbox": "click",
    "range": "modify_range",
    "text": "type",
    "select": "select",
    "textarea": "type",
    "hidden": "type",
    "crowd-checkbox": "click",
    "crowd-slider": "modify_range",
    "crowd-input": "type",
}
RESERVED_FIELDS = {"_id", "Task", "Title", "Description", "Keywords", "Template", "Answer"}
DYNAMICALLY_GENERATED_TASKS = {
    "Gun violence structured extraction",
    "Scalar Adjective Ordering",
    "Passive voice Parents 1st-2nd Person Persuasiveness Comparison",
    "TrecQA",
    "Simplicity rating",
    "Sentence Compression",
    "Scalar Adjectives Identification",
    "HTER",
    "HTER - longer sentences",
    "neural-pop (PLAN evaluation) t5-human-test b",
    "Paraphrase Clustering with Merge",
}
ELEMENT_CACHE: dict[str, dict[str, Any]] = {}
ERROR_MESSAGES: dict[str, int] = {}
PLAYWRIGHT_LOADER = None


def _print_error_once(message: str) -> None:
    ERROR_MESSAGES[message] = ERROR_MESSAGES.get(message, 0) + 1
    if ERROR_MESSAGES[message] == 1:
        print(message, file=sys.stderr)


def _html_snapshot(html_template: str) -> str:
    global PLAYWRIGHT_LOADER
    if PLAYWRIGHT_LOADER is None:
        from playwright.sync_api import sync_playwright

        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        PLAYWRIGHT_LOADER = (playwright, browser, page)
    _, _, page = PLAYWRIGHT_LOADER
    page.set_content(html_template)
    page.wait_for_load_state("networkidle")
    return page.content()


def _close_playwright() -> None:
    global PLAYWRIGHT_LOADER
    if PLAYWRIGHT_LOADER is None:
        return
    playwright, browser, page = PLAYWRIGHT_LOADER
    page.close()
    browser.close()
    playwright.stop()
    PLAYWRIGHT_LOADER = None


def _answer(record: dict[str, Any]) -> dict[str, str]:
    if isinstance(record.get("Answer"), dict):
        return {str(key): str(value) for key, value in record["Answer"].items()}
    return {
        key.split(".", 1)[1]: str(value)
        for key, value in record.items()
        if key.startswith("Answer.")
    }


def _filled_template(record: dict[str, Any]) -> tuple[str, bool]:
    html_template = str(record["Template"])
    use_cache = True
    for key, value in record.items():
        if key in RESERVED_FIELDS or key.startswith("Answer."):
            continue
        html_template = html_template.replace(f"${{{key}}}", str(value))
        if record["Task"] not in DYNAMICALLY_GENERATED_TASKS and " name=" in str(value):
            use_cache = False
    return html_template, use_cache


def _element_type(element: etree.ElementBase) -> str:
    if element.tag == "input":
        if element.get("type") in {
            "radio",
            "checkbox",
            "range",
            "hidden",
            "submit",
            "reset",
            "button",
        }:
            return str(element.get("type"))
        return "text"
    if element.tag in {"select", "textarea"}:
        return str(element.tag)
    if str(element.tag).startswith("crowd-"):
        return str(element.tag)
    return "text"


def _numeric_equal(first: str, second: str) -> bool:
    first = first.strip()
    second = second.strip()
    try:
        return float(first) == float(second)
    except ValueError:
        return first == second


def _verify_xpath(
    task: str, tree: etree.ElementBase, element: etree.ElementBase, xpath: str
) -> bool:
    try:
        result = tree.xpath(xpath)
    except Exception:
        _print_error_once(f"Invalid xpath: {xpath} in Task {task}")
        raise
    if not result:
        _print_error_once(f"Could not find element with xpath {xpath} in Task {task}")
    elif len(result) > 1:
        _print_error_once(f"Found multiple elements with xpath {xpath} in Task {task}")
    elif result[0] != element:
        _print_error_once(f"Element found with xpath does not match {xpath} in Task {task}")
    return bool(result and result[0] == element)


def _html_after_update(tree: etree.ElementBase, fake_url: str) -> Step:
    return web_observation_step(
        html=etree.tostring(tree).decode(),
        url=fake_url,
    )


def _input_elements(record: dict[str, Any], html_template: str, use_cache: bool):
    task = str(record["Task"])
    if use_cache and task in ELEMENT_CACHE:
        cached = ELEMENT_CACHE[task]
        return cached["_html_tree"], cached["_input_elements"], html_template

    if task in DYNAMICALLY_GENERATED_TASKS:
        html_template = _html_snapshot(html_template)
    tree = etree.HTML(html_template)
    input_elements = tree.xpath(ALL_INPUT_ELEMENTS_XPATH)
    ELEMENT_CACHE[task] = {"_html_tree": tree, "_input_elements": input_elements}
    return tree, input_elements, html_template


def _append_checkbox_or_radio(
    steps: list[Step],
    *,
    element: etree.ElementBase,
    tree: etree.ElementBase,
    value: str,
    task: str,
    fake_url: str,
) -> None:
    type_filter = f'and @type="{element.get("type")}"' if element.get("type") else ""
    value_filter = f'and @value="{element.get("value")}"' if element.get("value") else ""
    xpath = f'//{element.tag}[@name="{element.get("name")}" {type_filter} {value_filter}]'
    if not _verify_xpath(task, tree, element, xpath):
        return

    if not value and element.get("checked"):
        steps.append(tool_step("click", {"xpath": xpath}))
        del element.attrib["checked"]
        steps.append(_html_after_update(tree, fake_url))

    values_are_equal = _numeric_equal(value, element.get("value", "on"))
    if (
        not values_are_equal
        and _element_type(element) in {"checkbox", "crowd-checkbox"}
        and "|" in value
    ):
        values_are_equal = any(
            _numeric_equal(item, element.get("value", "on")) for item in value.split("|")
        )

    if value and not element.get("checked") and values_are_equal:
        steps.append(tool_step("click", {"xpath": xpath}))
        if _element_type(element) == "radio":
            for radio in tree.xpath(f"//input[@name='{element.get('name')}' and @type='radio']"):
                if radio.get("checked"):
                    del radio.attrib["checked"]
        element.attrib["checked"] = "checked"
        steps.append(_html_after_update(tree, fake_url))


def _append_range(
    steps: list[Step],
    *,
    element: etree.ElementBase,
    tree: etree.ElementBase,
    value: str,
    task: str,
    fake_url: str,
) -> None:
    if not value or _numeric_equal(value, element.get("value", "")):
        return
    type_filter = f'and @type="{element.get("type")}"' if element.get("type") else ""
    xpath = f'//{element.tag}[@name="{element.get("name")}" {type_filter}]'
    if _verify_xpath(task, tree, element, xpath):
        steps.append(tool_step("modify_range", {"xpath": xpath, "value": value}))
        element.attrib["value"] = value
        steps.append(_html_after_update(tree, fake_url))


def _append_select(
    steps: list[Step],
    *,
    element: etree.ElementBase,
    tree: etree.ElementBase,
    value: str,
    task: str,
    fake_url: str,
) -> None:
    xpath = f'//{element.tag}[@name="{element.get("name")}"]'
    if not _verify_xpath(task, tree, element, xpath):
        return
    if element.get("multiple") is not None:
        _print_error_once(f"Found select element with multiple attribute in Task {task}")
    options = element.xpath("./option")
    if any(
        option.get("selected") is not None
        and _numeric_equal(option.get("value", option.text or ""), value)
        for option in options
    ):
        return
    if (
        options
        and all(option.get("selected") is None for option in options)
        and _numeric_equal(
            options[0].get("value", options[0].text or ""),
            value,
        )
    ):
        return
    if any(_numeric_equal(option.get("value", option.text or ""), value) for option in options):
        steps.append(tool_step("select", {"xpath": xpath, "value": value}))
        for option in options:
            if _numeric_equal(option.get("value", option.text or ""), value):
                option.attrib["selected"] = "selected"
            elif option.get("selected"):
                del option.attrib["selected"]
        steps.append(_html_after_update(tree, fake_url))


def _append_text(
    steps: list[Step],
    *,
    element: etree.ElementBase,
    tree: etree.ElementBase,
    value: str,
    task: str,
    fake_url: str,
) -> None:
    type_filter = f'and @type="{element.get("type")}"' if element.get("type") else ""
    xpath = f'//{element.tag}[@name="{element.get("name")}" {type_filter}]'
    if not _verify_xpath(task, tree, element, xpath):
        return
    current_text = element.text or "" if element.tag == "textarea" else element.get("value", "")
    if _numeric_equal(current_text, value):
        return
    steps.append(tool_step("type", {"xpath": xpath, "value": value}))
    if element.tag == "textarea":
        element.text = value
    else:
        element.attrib["value"] = value
    steps.append(_html_after_update(tree, fake_url))


def convert_record(raw_record: dict[str, Any], dataset_name: str):
    answers = _answer(raw_record)
    html_template, use_cache = _filled_template(raw_record)
    tree, input_elements, html_template = _input_elements(raw_record, html_template, use_cache)
    fake_url = f"https://turkingbench.github.io/tasks/{urllib.parse.quote(str(raw_record['_id']))}"

    steps: list[Step] = [
        text_step(f"Go to {fake_url} and follow the instructions on the page", source="user"),
        tool_step("goto", {"url": fake_url}),
        web_observation_step(html=html_template, url=fake_url),
    ]
    task = str(raw_record["Task"])
    for element in input_elements:
        element_type = _element_type(element)
        name = element.get("name")
        if (
            element_type in {"hidden", "submit", "reset", "button"}
            or not name
            or name not in answers
        ):
            continue
        value = answers[name].strip()
        if element_type in {"checkbox", "crowd-checkbox", "radio"}:
            _append_checkbox_or_radio(
                steps,
                element=element,
                tree=tree,
                value=value,
                task=task,
                fake_url=fake_url,
            )
        elif element_type in {"range", "crowd-slider"}:
            _append_range(
                steps,
                element=element,
                tree=tree,
                value=value,
                task=task,
                fake_url=fake_url,
            )
        elif element_type == "select":
            _append_select(
                steps,
                element=element,
                tree=tree,
                value=value,
                task=task,
                fake_url=fake_url,
            )
        elif element_type in {"text", "textarea", "crowd-input"}:
            _append_text(
                steps,
                element=element,
                tree=tree,
                value=value,
                task=task,
                fake_url=fake_url,
            )
        else:
            _print_error_once(f"Unhandled input element type: {element_type}")

    return trajectory(
        dataset_name,
        str(raw_record["_id"]),
        steps,
        raw=raw_record,
        details={
            "task": raw_record.get("Task"),
            "title": raw_record.get("Title"),
            "task_description": raw_record.get("Description"),
            "keywords": raw_record.get("Keywords"),
        },
    )


def main(script_file: str) -> None:
    dataset_name = script_file.rsplit("/", 2)[-2]
    try:
        for line in sys.stdin:
            if line.strip():
                print(
                    convert_record(json.loads(line), dataset_name).model_dump_json(
                        exclude_none=True
                    )
                )
    finally:
        _close_playwright()
