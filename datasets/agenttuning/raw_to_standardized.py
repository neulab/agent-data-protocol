import json
import sys
sys.path.insert(1, './')
import re

from schema.action.action import Action
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

def convert_step(step: dict[str, str]) -> list[Action | Observation]:
    system_regex = re.match(
        r"(You are an assistant.*\n\nNow, my problem is:|Now, I will start a new problem in a new OS. My problem is:)\n\n(.*)",  # noqa
        step["content"],
        re.DOTALL,
    )
    code_act_regex = re.match(r"Think: (.*)\n\nAct: (.*)", step["content"], re.DOTALL)
    code_obs_regex = re.match(
        r"The output of the OS:\n(.*)", step["content"], re.DOTALL
    )
    if system_regex:
        return [
            TextObservation(content=system_regex.group(1), source="system"),
            TextObservation(content=system_regex.group(2), source="user"),
        ]
    elif code_act_regex:
        code_extract_regex = re.match(
            r"bash\n\n```bash\n(.*)\n```", code_act_regex.group(2), re.DOTALL
        )
        answer_extract_regex = re.match(
            r"answer\((.*)\)", code_act_regex.group(2), re.DOTALL
        )
        if code_extract_regex:
            return [
                CodeAction(
                    language="bash",
                    content=code_extract_regex.group(1),
                    description=code_act_regex.group(1),
                ),
            ]
        elif answer_extract_regex:
            return [
                MessageAction(
                    content=answer_extract_regex.group(1),
                    description=code_act_regex.group(1),
                ),
            ]
        else:
            raise ValueError(
                "Could not extract code from code action in"
                f" {json.dumps(step, indent=2)}"
            )
    elif code_obs_regex:
        return [
            TextObservation(content=code_obs_regex.group(1), source="os"),
        ]
    else:
        return [
            TextObservation(content=step["content"], source="user"),
        ]


for line in sys.stdin:
    raw_data = json.loads(line)

    content = []
    for step in raw_data["conversations"]:
        content.extend(convert_step(step))

   # Standardize the data
    standardize_data = Trajectory(
        id=raw_data["id"],
        content=content,
    )

    # Print the standardized data
    print(standardize_data.model_dump_json())
