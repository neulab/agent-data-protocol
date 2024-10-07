import sys
import json
from pathlib import Path
from typing import Any
from schema.action.api import ApiAction
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory
from schema_raw import SchemaRaw


download_instructions = """
    cd datasets/weblinx/
    git clone https://huggingface.co/datasets/McGill-NLP/WebLINX-full
    cd WebLINX-full/
    git lfs pull --exclude="candidates/*,chat/*,data/*,**/bboxes/*,*.mp4,*.png"
"""

ROLE_MAP = {
    "instructor": "user",
    "navigator": "assistant",
}

STEPS_TO_IGNORE = ["tabcreate", "tabswitch", "tabremove", "copy"]


weblinx_dump = Path(__file__).parent / "WebLINX-full"


def convert_step(step: Any, shortcode: str) -> list:
    if step.type == "chat":
        return [TextObservation(content=step.utterance, source=ROLE_MAP[step.speaker])]
    elif step.action["intent"] == "load":
        url = step.action["arguments"]["metadata"]["url"]
        return [ApiAction(function="goto", kwargs={"url": url})]
    elif step.action["intent"] in STEPS_TO_IGNORE:
        return []
    else:
        args = step.action["arguments"]
        web_observation = WebObservation(
            html=(weblinx_dump / "demonstrations" / shortcode / "pages"/ step.state.page).read_text(),
            url=args["metadata"]["url"],
            viewport_size=(args["metadata"]["viewportWidth"], args["metadata"]["viewportHeight"]),
            image_observation=None, # TODO: add image observation
        )
        _elid = args["element"]["attributes"].get("data-webtasks-id")
        xpath = f"//*[@data-webtasks-id='{_elid}']" if _elid else args["element"]["xpath"]
        if step.action["intent"] == "click":
            return [
                web_observation,
                ApiAction(
                    function="click",
                    kwargs={"xpath": xpath},
                )
            ]
        elif step.action["intent"] in ["textInput", "paste"]:
            value = args["text" if step.action["intent"] == "textInput" else "pasted"]
            return [
                web_observation,
                ApiAction(
                    function="type",
                    kwargs={"xpath": xpath, "value": value},
                )
            ]
        else:
            print(f"Unknown action: {step.action['intent']}", file=sys.stderr)
    return []


if __name__ == "__main__":

    assert weblinx_dump.is_dir(), "Please download the dataset first: " + download_instructions
    for line in sys.stdin:
        raw_data = json.loads(line)
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
        print(standardized_data.model_dump_json())