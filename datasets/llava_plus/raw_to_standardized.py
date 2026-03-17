import json
import os
import re
import sys

from schema.action.action import Action
from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.image import ImageObservation
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


def convert_step(step: dict[str, str], metadata) -> list[Action | Observation]:
    """Convert a single conversation step to standardized actions/observations."""
    if step["from"] == "human":
        text = step["value"]
        if "<image>" in text:
            # Strip all <image> occurrences and surrounding whitespace
            cleaned = re.sub(r"\s*<image>\s*", "", text).strip()
            result = [
                ImageObservation(
                    content=os.path.join("images/", metadata["data_source"], metadata["image"]),
                    annotations=None,
                    source="environment",
                ),
            ]
            if cleaned:
                result.append(TextObservation(content=cleaned, source="user"))
            return result
        else:
            return [TextObservation(content=text, source="user")]
    elif step["from"] == "gpt":
        if len(step["actions"]) > 0:
            content = [
                MessageAction(
                    content=step["value"],
                    description=step["thoughts"],
                )
            ]

            for act in step["actions"]:
                # Normalize API name: replace - and + with _ for valid Python identifiers
                api_name = act["API_name"].replace("-", "_").replace("+", "_")
                content.extend(
                    [
                        ApiAction(
                            function=api_name,
                            kwargs=act["API_params"],
                            description=None,
                        )
                    ]
                )

            return content
        else:
            return [
                MessageAction(
                    content=step["value"],
                    description=step["thoughts"],
                )
            ]
    else:
        raise Exception("Invalid role.")


# Process each line of input individually
for line in sys.stdin:
    raw_data = json.loads(line)
    metadata = dict([(k, v) for k, v in raw_data.items() if k != "conversations"])

    content = []
    for step in raw_data["conversations"]:
        content.extend(convert_step(step, metadata))

    # Standardize the data
    standardize_data = Trajectory(id=str(raw_data["unique_id"]), content=content)

    # Print the standardized data as JSON
    print(json.dumps(standardize_data.model_dump()))
