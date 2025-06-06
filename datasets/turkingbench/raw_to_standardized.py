import json
import sys
import urllib.parse

from lxml import etree
from playwright.sync_api import sync_playwright
from schema_raw import SchemaRaw

from schema.action.api import ApiAction
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory

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


RESERVED_FIELDS = set(["_id", "Task", "Title", "Description", "Keywords", "Template", "Answer"])
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

ELEMENT_CACHE = {}


def get_element_type(el: any) -> str:
    """
    Get the type of the input element

    Args:
        el: The input element

    Returns:
        The type of the input element (str)

    """
    if el.tag == "input":
        if el.get("type") in ["radio", "checkbox", "range"] + [
            "hidden",
            "submit",
            "reset",
            "button",
        ]:
            return el.get("type")
        else:
            return "text"
    elif el.tag in ["select", "textarea"]:
        return el.tag
    elif el.tag.startswith("crowd-"):
        return el.tag
    # maybe some custom input element
    # as long as it stores text in 'value' attribute, we can treat it as a text input
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
    try:
        res = html_tree.xpath(xpath)
    except Exception as e:
        print_error_once(f"Invalid xpath: {xpath} in Task {task}")
        raise e
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

    fake_url = f"https://turkingbench.github.io/tasks/{urllib.parse.quote(data['_id'])}"
    content: list = [
        TextObservation(
            content=f"Go to {fake_url} and follow the instructions on the page", source="user"
        ),
        ApiAction(function="goto", kwargs={"url": fake_url}),
        WebObservation(
            html=html_template,
            axtree=None,
            url=fake_url,
            viewport_size=None,
            image_observation=None,
        ),
    ]

    for el in input_elements:
        if (
            get_element_type(el) in ["hidden", "submit", "reset", "button"]
            or not el.get("name")
            or data["Answer"].get(el.get("name")) is None
        ):
            continue
        v = data["Answer"][el.get("name")].strip()
        if get_element_type(el) in ["checkbox", "crowd-checkbox", "radio"]:
            type_filter = f'and @type="{el.get("type")}"' if el.get("type") else ""
            value_filter = f'and @value="{el.get("value")}"' if el.get("value") else ""
            xpath = f'//{el.tag}[@name="{el.get("name")}" {type_filter} {value_filter}]'
            if not verify_xpath(data["Task"], tree, el, xpath):
                continue
            if not v and el.get("checked"):
                # this was a radio/checkbox that was initially checked
                # but no answer was recorded, that means we need to uncheck it
                content.append(ApiAction(function="click", kwargs={"xpath": xpath}))
                del el.attrib["checked"]
                content.append(
                    WebObservation(
                        html=etree.tostring(tree).decode(),
                        url=None,
                        viewport_size=None,
                        image_observation=None,
                    )
                )
            values_are_equal = numeric_equal(v, el.get("value", "on"))
            if (
                not values_are_equal
                and get_element_type(el) in ["checkbox", "crowd-checkbox"]
                and "|" in v
            ):
                # turkingbench represents multiple selected checkboxes with the same name
                # but different values as a single string of values separated by '|'
                values_are_equal = any(
                    [numeric_equal(value, el.get("value", "on")) for value in v.split("|")]
                )
            if v and not el.get("checked") and values_are_equal:
                # this was a radio/checkbox that was initially unchecked
                # but an answer was recorded, that means we need to check it
                content.append(ApiAction(function="click", kwargs={"xpath": xpath}))
                if get_element_type(el) == "radio":
                    # uncheck all other radios in the group
                    other_radios = tree.xpath(
                        f"//input[@name='{el.get('name')}' and @type='radio']"
                    )
                    for radio in other_radios:
                        if radio.get("checked"):
                            del radio.attrib["checked"]
                el.attrib["checked"] = "checked"
                content.append(
                    WebObservation(
                        html=etree.tostring(tree).decode(),
                        url=None,
                        viewport_size=None,
                        image_observation=None,
                    )
                )
        elif get_element_type(el) in ["range", "crowd-slider"]:
            if v and not numeric_equal(v, el.get("value", "")):
                type_filter = f'and @type="{el.get("type")}"' if el.get("type") else ""
                xpath = f'//{el.tag}[@name="{el.get("name")}" {type_filter}]'
                if not verify_xpath(data["Task"], tree, el, xpath):
                    continue
                content.append(
                    ApiAction(
                        function="modify_range",
                        kwargs={"xpath": xpath, "value": v},
                    )
                )
                el.attrib["value"] = v
                content.append(
                    WebObservation(
                        html=etree.tostring(tree).decode(),
                        url=None,
                        viewport_size=None,
                        image_observation=None,
                    )
                )
        elif get_element_type(el) == "select":
            xpath = f'//{el.tag}[@name="{el.get("name")}"]'
            if not verify_xpath(data["Task"], tree, el, xpath):
                continue
            if el.get("multiple") is not None:
                print_error_once(
                    f"Found select element with 'multiple' attribute in Task {data['Task']}:\n{etree.tostring(el).decode()}"
                )
            options = el.xpath("./option")
            # if the option is already selected, no need to select it again
            if any(
                [
                    o.get("selected") is not None and numeric_equal(o.get("value", o.text or ""), v)
                    for o in options
                ]
            ):
                continue
            # if no option has 'selected' attribute, that means the first option is selected by default
            if all([o.get("selected") is None for o in options]) and numeric_equal(
                options[0].get("value", options[0].text or ""), v
            ):
                continue
            if any([numeric_equal(o.get("value", o.text or ""), v) for o in options]):
                content.append(
                    ApiAction(
                        function="select",
                        kwargs={"xpath": xpath, "value": v},
                    )
                )
                # select the option and unselect all other options
                for option in options:
                    if numeric_equal(option.get("value", option.text or ""), v):
                        option.attrib["selected"] = "selected"
                    elif option.get("selected"):
                        del option.attrib["selected"]
                content.append(
                    WebObservation(
                        html=etree.tostring(tree).decode(),
                        url=None,
                        viewport_size=None,
                        image_observation=None,
                    )
                )
        elif get_element_type(el) in ["text", "textarea", "crowd-input"]:
            type_filter = f'and @type="{el.get("type")}"' if el.get("type") else ""
            xpath = f'//{el.tag}[@name="{el.get("name")}" {type_filter}]'
            if not verify_xpath(data["Task"], tree, el, xpath):
                continue
            text = el.text or "" if el.tag == "textarea" else el.get("value", "")
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
                content.append(
                    WebObservation(
                        html=etree.tostring(tree).decode(),
                        url=None,
                        viewport_size=None,
                        image_observation=None,
                    )
                )
        else:
            print_error_once(f"Unhandled input element type: {get_element_type(el)}")

    return Trajectory(
        id=data["_id"],
        content=content,
        details={
            "task": data["Task"],
            "title": data["Title"],
            "task_description": data["Description"],
            "keywords": data["Keywords"],
        },
    )


if __name__ == "__main__":
    # Process each line of input individually
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data).model_dump()
        standardized_data = process_data(data)

        # Print the standardized data as JSON
        print(json.dumps(standardized_data.model_dump()))
