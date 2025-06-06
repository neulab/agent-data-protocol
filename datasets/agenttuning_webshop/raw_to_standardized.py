import json
import random
import re
import sys
from typing import Tuple

from schema.action.action import Action
from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


def extract_thought_and_action(content: str) -> Tuple[str, str]:
    """
    Extract the thought and action from a message.
    """
    non_actions = [
        "task failed",
        "task succeeded",
        "end of the task",
        "end the task",
        "unable to perform the task",
        "task cannot be completed",
        "end task",
        "the task cannot be completed",
        "unable to complete the task",
    ]

    # match both THOUGHT and ACTION
    match = re.search(r"Thought:\n\s*(.*?)\s*Action:\n\s*(.*)", content, re.DOTALL)
    if match:
        thought = match.group(1).strip()
        action = match.group(2).strip()
        for non_action in non_actions:
            if non_action in action.lower():
                action = ""
        return thought, action

    # match ACTION only
    match = re.search(r"Action:\n\s*(.*)", content, re.IGNORECASE)
    if match:
        action = match.group(1).strip()
        for non_action in non_actions:
            if non_action in action.lower():
                action = ""
        return "", action

    # no action exists
    return "", ""


def format_code(raw_code: str) -> Tuple[str, str]:
    # Format the function name and argument list and add quotes
    match = re.match(r"(\w+)\[(.*)\]", raw_code)
    if not match:
        raise ValueError(f"Invalid function call format: {raw_code}")
    func_name = match.group(1)
    if func_name not in ["search", "click"]:
        raise ValueError(f"{raw_code} contains invalid function names")
    args_str = match.group(2)
    if not args_str:
        return func_name, {}
    # Since all apis only have one argument
    arg = "keywords" if func_name == "search" else "bid"
    args = {arg: f'"{args_str}"'}
    return func_name, args


def convert_step(step: dict[str, str]) -> list[Action | Observation]:
    system_regex = re.match(
        r"(You are web shopping.*)\n(.*)",  # noqa
        step["content"],
        re.DOTALL,
    )
    if system_regex:
        if "You are web shopping" in system_regex.group(1):
            system_prompt = system_regex.group(1).split(
                "You can use search action if search is available."
            )[0]
            return [
                TextObservation(content=system_prompt, source="system"),
            ]

    elif (
        "ok." == step["content"].strip().lower()
        or "ok. i'll follow" in step["content"].strip().lower()
    ):
        return []

    if step["role"] == "assistant":
        thought, action = extract_thought_and_action(step["content"])
        func_name, args = format_code(action)
        return [
            ApiAction(
                function=func_name,
                kwargs=args,
                description=thought,
            ),
        ]

    else:
        return [
            TextObservation(
                content=step["content"],
                source=step["role"],
            ),
        ]


for line in sys.stdin:
    raw_data = json.loads(line)
    content = []

    for step in raw_data["conversations"]:
        content.extend(convert_step(step))

    # Handle finish actions for natural language based tasks
    if not isinstance(content[-1], MessageAction) or "<finish>" not in content[-1].content:
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
                        content="<finish> </finish>",
                        description="I have successfully completed the task.",
                    ),
                ],
                [
                    MessageAction(
                        content="<finish> </finish>",
                        description="I did it! The task is now complete.",
                    ),
                ],
                [
                    MessageAction(
                        content="<finish> </finish>",
                        description="The objective has been achieved with no outstanding issues.",
                    ),
                ],
                [
                    MessageAction(
                        content="<finish> </finish>",
                        description="I have fulfilled all the requirements of the task.",
                    ),
                ],
                [
                    MessageAction(
                        content="<finish> </finish>",
                        description="I've wrapped up the task successfully.",
                    ),
                ],
            ]
        )
        content.extend(assistant_end_message)

    # Standardize the data
    standardize_data = Trajectory(
        id=raw_data["id"],
        content=content,
    )

    # Print the standardized data
    print(standardize_data.model_dump_json())
