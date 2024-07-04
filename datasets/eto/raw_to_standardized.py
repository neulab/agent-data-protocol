import json
import sys
import re

from schema.action.action import Action
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


def convert_example(example: dict[str, str]) -> list[Action | Observation]:
    content = [
        TextObservation(content=example["conversations"][0]["value"], source="system"),
        TextObservation(content=example["conversations"][2]["value"], source="user"),
    ]

    for i in range(3, len(example["conversations"]) - 1, 2):

        assert example["conversations"][i]["from"] == "gpt"
        assert example["conversations"][i + 1]["from"] == "human"

        thought_action_regex = re.match(
            r"Thought: (.*)\nAction: (.*)",
            example["conversations"][i]["value"],
            re.DOTALL,
        )

        content.append(
            MessageAction(
                content=thought_action_regex.group(2),
                description=thought_action_regex.group(1),
            )
        )

        observation_regex = re.match(
            r"Observation: (.*)", example["conversations"][i + 1]["value"], re.DOTALL
        )

        if observation_regex is None:
            print(example["conversations"][i + 1]["value"])

        content.append(
            TextObservation(
                content=observation_regex.group(1),
                source="environment",
            )
        )
    else:
        assert example["conversations"][-1]["from"] == "gpt"
        thought_action_regex = re.match(
            r"Thought: (.*)\nAction: (.*)",
            example["conversations"][-1]["value"],
            re.DOTALL,
        )
        content.append(
            MessageAction(
                content=thought_action_regex.group(2),
                description=thought_action_regex.group(1),
            )
        )

    return content


for line in sys.stdin:
    raw_example = json.loads(line)

    example = convert_example(raw_example)

    traj: Trajectory = Trajectory(
        id=raw_example["id"],
        content=example,
    )
    print(traj.model_dump_json())
