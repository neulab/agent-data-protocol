import json
import random
import re
import sys

from schema.action.action import Action
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

TOOL_DESCRIPTION = "Tool function available (already imported in <execute> environment):"
WARNING_MSG = "Observation:\nI don't understand your input. \nIf you want to execute code, please use <execute> YOUR_CODE_HERE </execute>.\nIf you want to give me an answer, please use <solution> YOUR_SOLUTION_HERE </solution>.\nFor example: The answer to the question is <solution> 42 </solution>."
NEW_WARNING_MSG = "I don't understand your input. \nIf you want to execute code, please follow the instructions.\nIf you want to give me an answer, please use <solution> YOUR_SOLUTION_HERE </solution>.\nFor example: The answer to the question is <solution> 42 </solution>."
APIS = set()


def convert_step(step: dict[str, str]) -> list[Action | Observation]:
    global APIS

    if step["role"] == "system":
        system_msg = step["content"]

        if TOOL_DESCRIPTION in system_msg:
            splited = system_msg.split(TOOL_DESCRIPTION, maxsplit=1)
            system_msg = splited[0].rstrip()
            APIS.add(splited[1])
        return []

    assert step["role"] in ["assistant", "user"], f"Invalid role: {step['role']}"
    task_regex = re.match(r"Task:\n(.*)", step["content"], re.DOTALL)
    solution_regex = re.match(r"(.*)<solution>(.*)</solution>", step["content"], re.DOTALL)
    execute_regex = re.match(r"(.*)<execute>(.*)</execute>", step["content"], re.DOTALL)
    obs_regex = re.match(r"Observation:\n(.*)", step["content"], re.DOTALL)

    if WARNING_MSG in step["content"]:
        return [
            TextObservation(
                content=step["content"].replace(WARNING_MSG, NEW_WARNING_MSG),
                source="environment"
                if step["role"] == "system"
                else "user"
                if step["role"] == "user"
                else "agent",
            ),
        ]
    elif solution_regex:
        assert step["role"] == "assistant", (
            f"Expected assistant role, got {step['role']}. {json.dumps(step, indent=2)}"
        )
        thought = solution_regex.group(1).strip()
        solution = solution_regex.group(2).strip()
        return [
            MessageAction(content=f"<solution> {solution} </solution>", description=thought or ""),
        ]

    elif execute_regex:
        assert step["role"] == "assistant", f"Expected assistant role, got {step['role']}"
        thought = execute_regex.group(1).strip()
        code = execute_regex.group(2).strip()
        assert code, f"Empty code in {json.dumps(step, indent=2)}"
        return [
            CodeAction(
                language="python",
                content=code,
                description=thought or "",
            ),
        ]

    elif obs_regex:
        return [
            TextObservation(
                content=obs_regex.group(1),
                source="environment"
                if step["role"] == "system"
                else "user"
                if step["role"] == "user"
                else "agent",
            ),
        ]

    elif task_regex:
        return [
            TextObservation(
                content=task_regex.group(1),
                source="environment"
                if step["role"] == "system"
                else "user"
                if step["role"] == "user"
                else "agent",
            ),
        ]

    else:
        # return raw messages directly if non of the regex matches
        return [
            MessageAction(content=step["content"], description=None),
        ]


# Process each line of input individually
for line in sys.stdin:
    raw_data = json.loads(line)
    content = []

    for step in raw_data["conversations"]:
        content.extend(convert_step(step))

    # append useful system prompt to first user message
    content[
        0
    ].content += "\n\nYou have 5 chances to interact with the environment or propose a solution. You can only propose a solution 2 times."

    if (isinstance(content[-1], TextObservation) and content[-1].source == "agent") or isinstance(
        content[-1], CodeAction
    ):
        user_end_message = random.choice(
            [
                [
                    TextObservation(
                        content="Congratulations! You have successfully solved the task.",
                        source="user",
                    ),
                ],
                [
                    TextObservation(
                        content="Your solution has been verified as correct. ", source="user"
                    ),
                ],
                [
                    TextObservation(
                        content="Well done on successfully completing the task!", source="user"
                    ),
                ],
                [
                    TextObservation(
                        content="Your implementation satisfies the task requirements.",
                        source="user",
                    ),
                ],
                [
                    TextObservation(content="Task completed successfully.", source="user"),
                ],
            ]
        )
        content.extend(user_end_message)
        assistant_end_message = random.choice(
            [
                [
                    MessageAction(
                        content="<finish> I have successfully completed the task. </finish>",
                        description="",
                    ),
                ],
                [
                    MessageAction(
                        content="<finish> I did it! The task is now complete. </finish>",
                        description="",
                    ),
                ],
                [
                    MessageAction(
                        content="<finish> The objective has been achieved with no outstanding issues. </finish>",
                        description="",
                    ),
                ],
                [
                    MessageAction(
                        content="<finish> I have fulfilled all the requirements of the task. </finish>",
                        description="",
                    ),
                ],
                [
                    MessageAction(
                        content="<finish> I've wrapped up the task successfully. </finish>",
                        description="",
                    ),
                ],
            ]
        )
        content.extend(assistant_end_message)

    # Handle finish actions for message actions
    if isinstance(content[-1], MessageAction) and "<finish>" not in content[-1].content:
        content[-1].content = f"<finish> {content[-1].content} </finish>"

    traj = Trajectory(
        id=raw_data["id"],
        content=content,
    )

    # Print the standardized data as JSON
    print(json.dumps(traj.model_dump()))

# with open("apis.txt", "w") as f:
#     for api in APIS:
#         f.write(api + "\n")
