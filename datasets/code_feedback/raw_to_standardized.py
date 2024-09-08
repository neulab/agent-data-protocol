import json
import sys
import re

from schema.action.action import Action
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


def convert_step(step: dict[str, str]) -> list[Action | Observation]:
    if step["role"] == "user":
        code_obs_regex = re.match(r'Execution result: \n(.*)', step["content"], re.DOTALL)

        if code_obs_regex:
            return [
                TextObservation(content=code_obs_regex.group(1), source="execution")
            ]
        else:
            return [
                TextObservation(content=step["content"], source="user")
            ]
    elif step["role"] == "assistant":
        code_extract_regex = re.match(r'(.*?)?```(\w+)?\n(.*?)\n```', step["content"], re.DOTALL)

        if code_extract_regex and code_extract_regex.group(2):
            return [
                CodeAction(
                    language=code_extract_regex.group(2).lower(),
                    content=code_extract_regex.group(3),
                    description=code_extract_regex.group(1),
                )
            ]
        else:
            return [
                MessageAction(
                    content=step["content"],
                    description=None,
                )
            ]
    else:
        raise Exception("Invalid role.")

for line in sys.stdin:
    raw_data = json.loads(line)

    content = []
    for step in raw_data["messages"]:
        content.extend(convert_step(step))

    # Standardize the data
    standardize_data = Trajectory(
        id=str(raw_data["id"]),
        content=content
    )

    # Print the standardized data
    print(standardize_data.model_dump_json())
