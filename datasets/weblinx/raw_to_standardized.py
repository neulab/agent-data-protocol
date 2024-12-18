import sys
import json
from pathlib import Path
from typing import Any
from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.observation.image import ImageObservation
from schema.trajectory import Trajectory
from schema_raw import SchemaRaw
from lxml import etree

not_found_count = 0
total_count = 0

DOWNLOAD_INSTRUCTIONS = """
    # Please download the raw dumps first:
    cd datasets/weblinx/
    git clone https://huggingface.co/datasets/McGill-NLP/WebLINX-full
    cd WebLINX-full/
    git lfs pull --exclude="candidates/*,chat/*,data/*,**/bboxes/*,*.mp4,*.png"
"""

INTENT_MAP = {
    "load": "goto",
    "click": "click",
    "textInput": "type",
    "paste": "type",
    "scroll": "scroll",
    "submit": "submit",
    "change": "select",
}

WEBLINX_DUMP = Path(__file__).parent / "WebLINX-full"

intents_skipped = set()


def xpath_exists(html: str, xpath: str) -> bool:
    """Check if the xpath exists in the html.

    Args:
    ----
        html (str): The html content.
        xpath (str): The xpath to check.

    """
    tree = etree.HTML(html)
    try:
        result = tree.xpath(xpath)
        return bool(result)
    except Exception:
        return False



def convert_step(
    step: Any, shortcode: str
) -> list[TextObservation | MessageAction | WebObservation | ApiAction]:
    """Convert a step in the raw data to a list of standardized actions.

    Args:
    ----
        step (Any): The step to convert.
        shortcode (str): The shortcode of the demonstration.

    """
    global not_found_count
    global total_count
    total_count += 1
    if step.type == "chat":
        if step.speaker == "instructor":
            return [TextObservation(content=step.utterance, source="user")]
        elif step.speaker == "navigator":
            return [MessageAction(content=step.utterance, description=None)]
        else:
            print(f"Unknown speaker: {step.speaker}", file=sys.stderr)
    elif step.action["intent"] == "load":
        return [
            ApiAction(
                function=INTENT_MAP[step.action["intent"]],
                kwargs={"url": step.action["arguments"]["metadata"]["url"]},
            )
        ]
    elif step.action["intent"] in INTENT_MAP:
        args = step.action["arguments"]
        image_observation = None
        if step.state.screenshot:
            img_path = (
                WEBLINX_DUMP / "demonstrations" / shortcode / "screenshots" / step.state.screenshot
            ).relative_to(Path.cwd()).as_posix()
            image_observation = ImageObservation(
                content=img_path, source="browser"
            )
        web_observation = WebObservation(
            html=(
                WEBLINX_DUMP / "demonstrations" / shortcode / "pages" / step.state.page
            ).read_text(),
            axtree=None,
            url=args["metadata"]["url"],
            viewport_size=(
                args["metadata"]["viewportWidth"],
                args["metadata"]["viewportHeight"],
            ),
            image_observation=image_observation,
        )
        if step.action["intent"] == "scroll":
            return [
                web_observation,
                ApiAction(
                    function=INTENT_MAP[step.action["intent"]],
                    kwargs={"dx": args["scrollX"], "dy": args["scrollY"]},
                ),
            ]
        _elid = args["element"]["attributes"].get("data-webtasks-id")
        if _elid:
            xpath = f"//*[@data-webtasks-id='{_elid}']"
            if not xpath_exists(web_observation.html, f"//*[@data-webtasks-id='{_elid}']"):
                _elid = None
        if not _elid:
            xpath = args["element"]["xpath"]
            if not xpath_exists(web_observation.html, xpath):
                xpath = "not found"
                not_found_count += 1
                
        if step.action["intent"] in ["click", "submit"]:
            return [
                web_observation,
                ApiAction(
                    function=INTENT_MAP[step.action["intent"]],
                    kwargs={"xpath": xpath},
                ),
            ]
        elif step.action["intent"] in ["textInput", "paste", "change"]:
            value_key = {
                "textInput": "text",
                "paste": "pasted",
                "change": "value",
            }
            value = args[value_key[step.action["intent"]]]
            return [
                web_observation,
                ApiAction(
                    function=INTENT_MAP[step.action["intent"]],
                    kwargs={"xpath": xpath, "value": value},
                ),
            ]
        else:
            print(f"Unknown intent: {step.action['intent']}", file=sys.stderr)
    else:
        intents_skipped.add(step.action["intent"])
    return []


if __name__ == "__main__":
    assert WEBLINX_DUMP.is_dir(), DOWNLOAD_INSTRUCTIONS
    shortcode_errors = []
    for line in sys.stdin:
        try:
            raw_data = json.loads(line)
            data = SchemaRaw(**raw_data)

            content: list = []
            for step in data.data:
                content.extend(convert_step(step, data.shortcode))

            standardized_data = Trajectory(
                id=data.shortcode,
                content=content,
                details={
                    "task_description": data.description,
                    "tasks": ", ".join(data.tasks),
                },
            )
            print(standardized_data.model_dump_json())

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            shortcode_errors.extend([{'shortcode': raw_data['shortcode'], 'error': str(e)}])
            with open("datasets/weblinx/shortcode_errors.json", "w") as f:
                json.dump(shortcode_errors, f, indent=4)
            continue
    print("intents skipped: " + ", ".join(intents_skipped), file=sys.stderr)
    print(f"not found: {not_found_count}/{total_count}", file=sys.stderr)