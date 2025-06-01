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


def convert_step(step: dict[str, str]) -> list[Action | Observation]:
    if step["role"] == "user":
        code_obs_regex = re.match(r"Execution result: \n(.*)", step["content"], re.DOTALL)

        if code_obs_regex:
            return [TextObservation(content=code_obs_regex.group(1), source="execution")]
        else:
            return [TextObservation(content=step["content"], source="user")]
    elif step["role"] == "assistant":
        code_extract_regex = re.match(r"(.*?)?```(\w+)?\n(.*?)\n```", step["content"], re.DOTALL)

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

    # Handle finish actions for natural language based tasks and code actions
    if (
        isinstance(content[-1], TextObservation)
        and content[-1].source == "assistant"
        or isinstance(content[-1], CodeAction)
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

    # Standardize the data
    standardize_data = Trajectory(id=str(raw_data["id"]), content=content)

    # Print the standardized data
    print(standardize_data.model_dump_json())
