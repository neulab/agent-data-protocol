import json
import sys
import re

from schema.action.action import Action
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


TOOL_DESCRIPTION = "Tool function available (already imported in <execute> environment):"
WARNING_MSG = "Observation:\nI don't understand your input. \nIf you want to execute code, please use <execute> YOUR_CODE_HERE </execute>.\nIf you want to give me an answer, please use <solution> YOUR_SOLUTION_HERE </solution>.\nFor example: The answer to the question is <solution> 42 </solution>."

def convert_step(step: dict[str, str]) -> list[Action | Observation]:
    global APIS

    if step["role"] == "system":
        system_msg = step["content"]
        
        if TOOL_DESCRIPTION in system_msg:
            splited = system_msg.split(TOOL_DESCRIPTION, maxsplit=1)
            system_msg = splited[0].rstrip()
            APIS.add(splited[1])

        return [
            TextObservation(content=system_msg, source=step["role"]),
        ]

    assert step["role"] in ["assistant", "user"], f"Invalid role: {step['role']}"

    task_regex = re.match(r"Task:\n(.*)", step["content"], re.DOTALL)
    solution_regex = re.match(r"(.*)<solution>(.*)</solution>", step["content"], re.DOTALL)
    execute_regex = re.match(r"(.*)<execute>(.*)</execute>", step["content"], re.DOTALL)
    obs_regex = re.match(r"Observation:\n(.*)", step["content"], re.DOTALL)

    if WARNING_MSG in step["content"]:
        return [
            TextObservation(content=step["content"], source=step["role"]),
        ]
    elif solution_regex:
        assert step["role"] == "assistant", f"Expected assistant role, got {step['role']}. {json.dumps(step, indent=2)}"
        thought = solution_regex.group(1).strip()
        solution = solution_regex.group(2).strip()
        if "<execute>" not in thought:
            return [
                MessageAction(content=solution, description=thought),
            ]
        else:
            # some of the thoughts contains <execute> tag which could be confusing
            # to model for training, so i did not include thought for solution
            return [
                MessageAction(content=solution, description=None),
            ]

    elif execute_regex:
        assert step["role"] == "assistant", f"Expected assistant role, got {step['role']}"
        thought = execute_regex.group(1).strip()
        code = execute_regex.group(2).strip()
        assert code, f"Empty code in {json.dumps(step, indent=2)}"
        return [
            CodeAction(
                language="python3",
                content=code,
                description=thought if thought else None,
            ),
        ]

    elif obs_regex:
        return [
            TextObservation(content=obs_regex.group(1), source=step["role"]),
        ]

    elif task_regex:
        return [
            TextObservation(content=task_regex.group(1), source=step["role"]),
        ]

    else:
        # return raw messages directly if non of the regex matches
        return [
            MessageAction(content=step["content"], description=None),
        ]

APIS = set()

for line in sys.stdin:
    raw_data = json.loads(line)

    content = []
    for step in raw_data["conversations"]:
        content.extend(convert_step(step))

    traj: Trajectory = Trajectory(
        id=raw_data["id"],
        content=content,
    )
    print(traj.model_dump_json())

# with open("apis.txt", "w") as f:
#     for api in APIS:
#         f.write(api + "\n")
