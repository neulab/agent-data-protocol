import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

from schema_raw import SchemaRaw

from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.image import ImageObservation
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory

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


def convert_step(
    step: Any, shortcode: str
) -> list[Union[TextObservation, MessageAction, WebObservation, ApiAction]]:
    """Convert a step in the raw data to a list of standardized actions.

    Args:
    ----
        step (Any): The step to convert.
        shortcode (str): The shortcode of the demonstration.

    """
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
                (
                    WEBLINX_DUMP
                    / "demonstrations"
                    / shortcode
                    / "screenshots"
                    / step.state.screenshot
                )
                .relative_to(Path.cwd())
                .as_posix()
            )
            image_observation = ImageObservation(content=img_path, source="environment")
        web_observation = WebObservation(
            html=(
                WEBLINX_DUMP / "demonstrations" / shortcode / "pages" / step.state.page
            ).read_text(),
            url=args["metadata"]["url"],
            viewport_size=[
                args["metadata"]["viewportWidth"],
                args["metadata"]["viewportHeight"],
            ],
            image_observation=image_observation,
            axtree=None,
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
        xpath = f"//*[@data-webtasks-id='{_elid}']" if _elid else args["element"]["xpath"]
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


def process_data(raw_data_list: List[Dict]) -> List[Dict]:
    """Process a list of raw data into standardized trajectories.

    Args:
        raw_data_list: List of raw data dictionaries

    Returns:
        List of standardized trajectory dictionaries
    """
    standardized_trajectories = []

    for raw_data in raw_data_list:
        data = SchemaRaw(**raw_data)

        content: list = []
        for step in data.data:
            content.extend(convert_step(step, data.shortcode))

        standardized_data = Trajectory(
            id=data.shortcode,
            content=content,
            details={
                "description": data.description,
                "tasks": ", ".join(data.tasks),
            },
        )
        standardized_trajectories.append(standardized_data.model_dump())

    return standardized_trajectories


def create_sample_std():
    """Create a sample standardized trajectory for testing purposes."""
    sample_trajectory = {
        "id": "sample-weblinx-trajectory",
        "content": [
            {
                "class_": "text_observation",
                "content": "I need help with booking a flight",
                "source": "user",
            },
            {
                "class_": "message_action",
                "content": "I'll help you book a flight. Let me navigate to a travel website.",
                "description": None,
            },
            {
                "class_": "api_action",
                "function": "goto",
                "kwargs": {"url": "https://example.com/flights"},
                "description": None,
            },
            {
                "class_": "web_observation",
                "html": "<html><body>Flight booking page</body></html>",
                "url": "https://example.com/flights",
                "viewport_size": [1024, 768],
                "image_observation": {
                    "class_": "image_observation",
                    "content": "datasets/weblinx/sample_screenshot.png",
                    "source": "environment",
                    "annotations": [],
                },
                "axtree": None,
            },
            {
                "class_": "api_action",
                "function": "type",
                "kwargs": {"xpath": "//input[@id='departure']", "value": "New York"},
                "description": None,
            },
        ],
    }
    return [sample_trajectory]


def process_single_data(raw_data: Dict) -> Dict:
    """Process a single raw data into a standardized trajectory.

    Args:
        raw_data: Raw data dictionary

    Returns:
        Standardized trajectory dictionary
    """
    data = SchemaRaw(**raw_data)

    content: list = []
    for step in data.data:
        content.extend(convert_step(step, data.shortcode))

    standardized_data = Trajectory(
        id=data.shortcode,
        content=content,
        details={
            "description": data.description,
            "tasks": ", ".join(data.tasks),
        },
    )
    return standardized_data.model_dump()


if __name__ == "__main__":
    # Check if WebLINX-full directory exists
    if not WEBLINX_DUMP.is_dir():
        print(
            f"Warning: {WEBLINX_DUMP} directory not found. Using sample data instead.",
            file=sys.stderr,
        )
        print(f"{DOWNLOAD_INSTRUCTIONS}", file=sys.stderr)
        # Create a sample standardized trajectory for testing
        standardized_trajectories = create_sample_std()

        # Print the standardized data as JSONL (one JSON object per line)
        for traj in standardized_trajectories:
            print(json.dumps(traj))
    else:
        # Process each line of input individually
        for line in sys.stdin:
            raw_data = json.loads(line)
            standardized_data = process_single_data(raw_data)

            # Print the standardized data as JSON
            print(json.dumps(standardized_data))

        if intents_skipped:
            print("intents skipped: " + ", ".join(intents_skipped), file=sys.stderr)
