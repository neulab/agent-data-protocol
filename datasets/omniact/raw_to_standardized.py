import json
import re
import sys

from schema.action.action import Action
from schema.action.code import CodeAction
from schema.observation.image import BoundingBox, ImageAnnotation, ImageObservation
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


def convert_example(example: dict[str, str]) -> list[Action | Observation]:
    task_regex = re.match(
        r"Task: (.*)\nOutput Script:\n(.*)|Task: (.*)\nOutput Script: \n(.*)|Task: (.*)\nOutput Script: (.*)|Task: (.*)\n(.*)\n|Task: (.*)\n(.*)",
        example["task"],
        re.DOTALL,
    )
    try:
        annotations = [
            ImageAnnotation(
                text=example["box"]["label"][index],
                element_type="button",
                bounding_box=BoundingBox(
                    x=example["box"]["top_left"][index][0],
                    y=example["box"]["top_left"][index][1],
                    width=example["box"]["bottom_right"][index][0]
                    - example["box"]["top_left"][index][0],
                    height=example["box"]["bottom_right"][index][1]
                    - example["box"]["top_left"][index][1],
                ),
            )
            for index in range(len(example["box"]["label"]))
        ]
    except:
        raise Exception(f"Error in example: {example['id'], example['task'], example['box']}")

    try:
        return [
            ImageObservation(
                content=example["image"],
                annotations=annotations,
                source="environment",
            ),
            TextObservation(
                content=task_regex.group(1)
                or task_regex.group(3)
                or task_regex.group(5)
                or task_regex.group(7)
                or task_regex.group(9),
                source="user",
            ),
            CodeAction(
                language="python",
                content=task_regex.group(2)
                or task_regex.group(4)
                or task_regex.group(6)
                or task_regex.group(8)
                or task_regex.group(10),
                description="Output Script",
            ),
        ]
    except:
        raise Exception(
            f"Error in example: {example['id'], example['task'], task_regex}"
        )  # task_regex.group(9), task_regex.group(10)


for line in sys.stdin:
    raw_example = json.loads(line)
    example = convert_example(raw_example)

    traj: Trajectory = Trajectory(
        id=raw_example["id"],
        content=example,
    )
    print(traj.model_dump_json())
