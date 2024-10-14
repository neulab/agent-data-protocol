import os
import sys
import json

from schema.action.api import ApiAction
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory
from schema_raw import SchemaRaw
from collections import defaultdict
from playwright.sync_api import sync_playwright
from lxml import etree



INPUT_ELEMENTS = [
    "input",
    "textarea",
    "select",
    "crowd-checkbox",
    "crowd-slider",
    "crowd-input",
]
ALL_INPUT_ELEMENTS_XPATH = " | ".join(f"//{el}" for el in INPUT_ELEMENTS)

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


RESERVED_FIELDS = set(
    ["_id", "Task", "Title", "Description", "Keywords", "Template", "Answer"]
)
NULL_VALUES = set(["", "none", "null", "na", "n/a", "no", "empty", "false"])

DYNAMICALLY_GENERATED_TASKS = set(
    [
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
    ]
)

MISSING_COUNTS = defaultdict(lambda: defaultdict(int))
ELEMENT_CACHE = {}


def get_element_type(el: any) -> str:
    """
    Get the type of the input element

    Args:
        el: The input element

    Returns:
        The type of the input element (str)

    """
    if el.get("type"):
        return el.get("type")
    elif el.tag == "select":
        return "select"
    elif el.tag.startswith("crowd-"):
        return el.tag
    return "text"


ERROR_MESSAGES = {}


def print_error_once(err_msg: str) -> None:
    """
    Print the error message only once

    Args:
        err_msg: The error message

    """
    if err_msg not in ERROR_MESSAGES:
        print(err_msg, file=sys.stderr)
        ERROR_MESSAGES[err_msg] = 1
    else:
        ERROR_MESSAGES[err_msg] += 1


class PlaywrightLoader:
    def __init__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.page = self.browser.new_page()

    def get_html_snapshot(self, html_template: str) -> str:
        """
        Fetch dynamic content using Playwright

        Args:
            html_template: The HTML template

        Returns:
            The modified html (str)
        """
        self.page.set_content(html_template)
        self.page.wait_for_load_state("networkidle")
        html = self.page.content()
        return html

    def close(self):
        self.page.close()
        self.browser.close()
        self.playwright.stop()

playwright_loader = PlaywrightLoader()


def numeric_equal(a: str, b: str) -> bool:
    """
    Check if two strings are numerically equal, otherwise check normal equality.
    Need to use this because turkingbench dataset sometimes represents 1 as 1.0 etc.

    Args:
        a: The first string
        b: The second string

    """
    a, b = a.strip(), b.strip()
    try:
        return float(a) == float(b)
    except ValueError:
        return a == b


def verify_xpath(task: str, html_tree: etree.Element, el: etree.Element, xpath: str) -> bool:
    """
    Verify if the xpath is correct

    Args:
        task: The task
        html_tree: The HTML tree
        el: The element
        xpath: The xpath

    Returns:
        Whether the xpath is correct (bool)

    """
    res = html_tree.xpath(xpath)
    if not res:
        print_error_once(f"Could not find element with xpath {xpath} in Task {task}")
    elif len(res) > 1:
        print_error_once(f"Found multiple elements with xpath {xpath} in Task {task}")
    elif res[0] != el:
        print_error_once(f"Element found with xpath does not match {xpath} in Task {task}")
    return res and res[0] == el


def process_data(data: dict) -> Trajectory:
    """
    Process the data

    Args:
        data: The raw data (dict)

    Returns:
        The standardized data (Trajectory)

    """
    html_template = data["Template"]
    use_cache = True
    for key in data:
        if key in RESERVED_FIELDS:
            continue
        # replaces '${col_name}' in html_template with raw_data["col_name"]
        html_template = html_template.replace(f"${{{key}}}", data[key])
        if data["Task"] not in DYNAMICALLY_GENERATED_TASKS and " name=" in data[key]:
            # sometimes there are html snippets in these batch.csv columns
            # In that case, we should not use soup_cache
            # but if it's a dynamically generated html template that requires playwright
            # we should use soup_cache, extraction will be very slow
            # (the 11 dynamically_generated_tasks don't have html snippets in batch.csv anyway)
            use_cache = False

    content: list = [
        WebObservation(
            html=html_template, url=None, viewport_size=None, image_observation=None
        )
    ]

    if use_cache and data["Task"] in ELEMENT_CACHE:
        tree = ELEMENT_CACHE[data["Task"]]["_html_tree"]
        input_elements = ELEMENT_CACHE[data["Task"]]["_input_elements"]
    else:
        if data["Task"] in DYNAMICALLY_GENERATED_TASKS:
            # the html_template has javascript that dynamically generates input elements
            # use playwright to run the javascript and get the modified html
            # Doesn't take care of all cases, for example if number of input elements changes based on user input
            html_template = playwright_loader.get_html_snapshot(html_template)
        tree = etree.HTML(html_template)
        input_elements = tree.xpath(ALL_INPUT_ELEMENTS_XPATH)
        ELEMENT_CACHE[data["Task"]] = {
            "_html_tree": tree,
            "_input_elements": input_elements,
        }

    for el in input_elements:
        if (
            get_element_type(el) == "hidden"
            or not el.get("name")
            or data["Answer"].get(el.get("name")) is None
        ):
            continue
        v = data["Answer"][el.get("name")].strip()
        if get_element_type(el) in ["checkbox", "crowd-checkbox", "radio"]:
            if el.get("value"):
                xpath = f"//{el.tag}[@name='{el.get('name')}' and @type='{el.get('type')}' and @value='{el.get('value')}]"
            else:
                xpath = f"//{el.tag}[@name='{el.get('name')}' and @type='{el.get('type')}]"
            if not verify_xpath(data["Task"], tree, el, xpath):
                continue
            if not v and el.get("checked"):
                # this was a radio/checkbox that was initially checked
                # but no answer was recorded, that means we need to uncheck it
                content.append(ApiAction(function="click", kwargs={"xpath": xpath}))
                del el.attrib["checked"]
                content.append(WebObservation(html=etree.tostring(tree).decode(), url=None, viewport_size=None, image_observation=None))
            if v and not el.get("checked") and numeric_equal(v, el.get("value", "on")):
                # this was a radio/checkbox that was initially unchecked
                # but an answer was recorded, that means we need to check it
                content.append(ApiAction(function="click", kwargs={"xpath": xpath}))
                el.attrib["checked"] = "checked"
                if get_element_type(el) == "radio":
                    # uncheck all other radios in the group
                    other_radios = tree.xpath(f"//input[@name='{el.get('name')}' and @type='radio']")
                    for radio in other_radios:
                        if radio.get("checked"):
                            del radio.attrib["checked"]
                content.append(WebObservation(html=etree.tostring(tree).decode(), url=None, viewport_size=None, image_observation=None))
        elif get_element_type(el) in ["range", "crowd-slider"]:
            xpath = f"//{el.tag}[@name='{el.get('name')}' and @type='{el.get('type')}']" if el.get("type") else f"//{el.tag}[@name='{el.get('name')}']"
            if not verify_xpath(data["Task"], tree, el, xpath):
                continue
            if v and not numeric_equal(v, el["value"]):
                content.append(
                    ApiAction(
                        function="modify_range",
                        kwargs={"xpath": xpath, "value": v},
                    )
                )
                el.attrib["value"] = v
                content.append(WebObservation(html=etree.tostring(tree).decode(), url=None, viewport_size=None, image_observation=None))
        elif get_element_type(el) == "select":
            xpath = f"//{el.tag}[@name='{el.get('name')}]"
            if not verify_xpath(data["Task"], tree, el, xpath):
                continue
            if el.get("multiple"):
                print_error_once(f"Found multiple select element in Task {data['Task']}:\n{etree.tostring(el).decode()}")
            options = el.xpath("./option")
            # if the option is already selected, no need to select it again
            if any([o.get("selected") and numeric_equal(o.get("value", o.text), v) for o in options]):
                continue
            # if the first option is selected by default and it's value is equal to the answer, no need to select it again
            if numeric_equal(options[0].get("value", options[0].text), v) and all([o.get("selected") is None for o in options]):
                continue
            if any([numeric_equal(o.get("value", o.text), v) for o in options]):
                content.append(
                    ApiAction(
                        function="select",
                        kwargs={"xpath": xpath, "value": v},
                    )
                )
                for option in options:
                    if numeric_equal(option.get("value", option.text), v):
                        option.attrib["selected"] = "selected"
                    elif option.get("selected"):
                        del option.attrib["selected"]
                content.append(WebObservation(html=etree.tostring(tree).decode(), url=None, viewport_size=None, image_observation=None))
        elif get_element_type(el) in ["text", "textarea", "crowd-input"]:
            xpath = f"//{el.tag}[@name='{el.get('name')}' and @type='{el.get('type')}']" if el.get("type") else f"//{el.tag}[@name='{el.get('name')}']"
            if not verify_xpath(data["Task"], tree, el, xpath):
                continue
            text = el.text if el.tag == "textarea" else el.get("value")
            if not numeric_equal(text, v):
                content.append(
                    ApiAction(
                        function="type",
                        kwargs={"xpath": xpath, "value": v},
                    )
                )
                if el.tag == "textarea":
                    el.text = v
                else:
                    el.attrib["value"] = v
                content.append(WebObservation(html=etree.tostring(tree).decode(), url=None, viewport_size=None, image_observation=None))
        else:
            print_error_once(f"Unhandled input element type: {get_element_type(el)}")

    return Trajectory(
        id=data["_id"],
        content=content,
        details={
            "task": data["Task"],
            "title": data["Title"],
            "description": data["Description"],
            "keywords": data["Keywords"],
        },
    )


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data).model_dump()
        standardized_data = process_data(data)
        print(standardized_data.model_dump_json())

    if "DEBUG" in os.environ:
        for task, counts in MISSING_COUNTS.items():
            for name, count in counts.items():
                print(f"{task}\t{name}\t{count}", file=sys.stderr)
