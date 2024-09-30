import os
import sys
import json

from schema.action.api import ApiAction
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory
from schema_raw import SchemaRaw
from bs4 import BeautifulSoup
from collections import defaultdict
from playwright.sync_api import sync_playwright


INPUT_ELEMENTS = [
    "input",
    "textarea",
    "select",
    "crowd-checkbox",
    "crowd-slider",
    "crowd-input",
]

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
SOUP_CACHE = {}


def get_element_type(el: any) -> str:
    """
    Get the type of the input element

    Args:
        el: The input element

    Returns:
        The type of the input element (str)

    """
    if el.get("type"):
        return el["type"]
    elif el.name == "select":
        return "select"
    elif el.name.startswith("crowd-"):
        return el.name
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


def fetch_dynamic_content(html_template: str) -> str:
    """
    Fetch dynamic content using Playwright

    Args:
        html_template: The HTML template

    Returns:
        The modified html (str)

    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(html_template)
        page.wait_for_load_state("networkidle")
        html = page.content()
        browser.close()
        return html


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
        if (
            data["Task"] not in DYNAMICALLY_GENERATED_TASKS
            and " name=" in data[key]
        ):
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

    if use_cache and data["Task"] in SOUP_CACHE:
        soup = SOUP_CACHE[data["Task"]]["_beautiful_soup"]
    else:
        if data["Task"] in DYNAMICALLY_GENERATED_TASKS:
            # the html_template has javascript that dynamically generates input elements
            # use playwright to run the javascript and get the modified html
            # Doesn't take care of all cases, for example if number of input elements changes based on user input
            html_template = fetch_dynamic_content(html_template)
        soup = BeautifulSoup(html_template, "html.parser")
        SOUP_CACHE[data["Task"]] = {"_beautiful_soup": soup}

    for k, v in data["Answer"].items():
        if use_cache and k in SOUP_CACHE[data["Task"]]:
            input_element = SOUP_CACHE[data["Task"]][k]
        else:
            input_element = soup.find_all(INPUT_ELEMENTS, {"name": k}, limit=1)
            SOUP_CACHE[data["Task"]][k] = input_element

        if input_element:
            el = input_element[0]
            input_type = get_element_type(el)
            action = ACTIONS.get(input_type)
            if not action:
                if "DEBUG" in os.environ:
                    print_error_once(
                        f"Could not find action for input element: type {input_type}, name {k}\n"
                        f"  Task: {data['Task']}\n"
                        f"  Title: {data['Title']}"
                    )
                continue
            if input_type == "hidden":
                continue
            xpath = f"//{el.name}[@name='{k}']" if not el.get("type") else f"//{el.name}[@name='{k}' and @type='{el["type"]}']"
            kwargs={"xpath": xpath}
            if action != "click":
                kwargs["value"] = v.strip()
            api_action = ApiAction(function=action, kwargs=kwargs)
            content.append(api_action)
        else:
            # we can ignore "col_name.value" if a corresponding "col_name" is present
            # github.com/JHU-CLSP/turking-bench/blob/5c842ada548a919b9d6130f628c8f30f7e8c8eac/src/utils/clean_csv.py#L55
            if "." in k and k.rsplit(".", 1)[0] in data["Answer"]:
                continue
            # turkingbench sometimes autogenerates Answer.input_name_{n} not all of which may be present in the html
            # log missing error only if the value is not null
            if v.strip().lower() not in NULL_VALUES:
                MISSING_COUNTS[data["Task"]][k] += 1
            continue

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