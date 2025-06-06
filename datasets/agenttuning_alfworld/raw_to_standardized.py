import json
import random
import re
import sys
from typing import Tuple

from schema.action.action import Action
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


def convert_system(system_regex: re.Match[str]) -> list[Observation]:
    assert "Interact with a household to solve a task" in system_regex.group(1)
    system_prompt = (
        system_regex.group(1).split("You should choose from two actions:")[0].strip() + "\n"
    )
    system_prompt += (
        system_regex.group(1)
        .split('Your output must strictly follow this format:"ACTION: your next action\n". ')[-1]
        .strip()
    )
    return [
        TextObservation(content=system_prompt, source="system"),
        TextObservation(content="Ok? Understood?", source="user"),
    ]


def parse_action(action_str: str) -> str:
    """
    Convert natural language ACTION strings into function calls.
    """
    action_str = action_str.strip().lower()

    # Match "go to X"
    match = re.match(r"go to (.+)", action_str)
    if match:
        return f"go('{match.group(1)}')"

    # Match "take X from Y"
    match = re.match(r"take (.+) from (.+)", action_str)
    if match:
        return f"take('{match.group(1)}', '{match.group(2)}')"

    # Match "put X in/on Y"
    match = re.match(r"put (.+) in/on (.+)", action_str)
    if match:
        return f"put('{match.group(1)}', '{match.group(2)}')"

    # Match "put X on Y"
    match = re.match(r"put (.+) on (.+)", action_str)
    if match:
        return f"put('{match.group(1)}', '{match.group(2)}')"

    # Match "open X"
    match = re.match(r"open (.+)", action_str)
    if match:
        return f"open('{match.group(1)}')"

    # Match "close X"
    match = re.match(r"close (.+)", action_str)
    if match:
        return f"close('{match.group(1)}')"

    # Match "heat X using Y"
    match = re.match(r"heat (.+) with (.+)", action_str)
    if match:
        return f"heat('{match.group(1)}', '{match.group(2)}')"

    # Match "examine X"
    match = re.match(r"examine (.+)", action_str)
    if match:
        return f"examine('{match.group(1)}')"

    # Match "cool X with Y"
    match = re.match(r"cool (.+) with (.+)", action_str)
    if match:
        return f"cool('{match.group(1)}', '{match.group(2)}')"

    # Match "use X"
    match = re.match(r"use (.+)", action_str)
    if match:
        return f"use('{match.group(1)}')"

    # Match "clean X with Y"
    match = re.match(r"clean (.+) with (.+)", action_str)
    if match:
        return f"clean('{match.group(1)}', '{match.group(2)}')"

    # Match "report problem with X"
    match = re.match(r"report problem with (.+)", action_str)
    if match:
        return f"report_problem('{match.group(1)}')"

    # Match "inventory"
    if action_str == "inventory":
        return "inventory()"

    # Match "look"
    if action_str == "look":
        return "look()"

    # Match "look at X under Y"
    match = re.match(r"look at (.+) under (.+)", action_str)
    if match:
        return f"look_at_under('{match.group(1)}', '{match.group(2)}')"

    raise ValueError(f"Unrecognized ACTION format: {action_str}")


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
    match = re.search(r"THOUGHT:\s*(.*?)\s*ACTION:\s*(.*)", content, re.DOTALL)
    if match:
        thought = match.group(1).strip()
        action = match.group(2).strip()
        for non_action in non_actions:
            if non_action in action.lower():
                action = ""
        return thought, action

    # match ACTION only
    match = re.search(r"ACTION:\s*(.*)", content, re.IGNORECASE)
    if match:
        action = match.group(1).strip()
        for non_action in non_actions:
            if non_action in action.lower():
                action = ""
        return "", action

    # no action exists
    return "", ""


def convert_step(step: dict[str, str]) -> list[Action | Observation]:
    # parse system prompt
    system_regex = re.match(
        r"(Interact with a household to solve a task.*)",  # noqa
        step["content"],
        re.DOTALL,
    )
    if system_regex:
        return convert_system(system_regex)

    # parse special message
    elif (
        "ok" == step["content"].strip().lower()
        or "ok. i'll follow" in step["content"].strip().lower()
    ):
        return [
            MessageAction(content=step["content"]),
        ]

    elif step["role"] == "assistant":
        thought, action = extract_thought_and_action(step["content"])
        if action:
            return [
                CodeAction(
                    language="python",
                    content=parse_action(action),
                    description=thought,
                ),
            ]
        else:
            return [
                TextObservation(content=thought if thought else step["content"], source="assistant")
            ]

    else:
        return [
            TextObservation(
                content=step["content"].replace("ACTION: ", ""),
                source=step["role"] if step["role"] != "system" else "user",
            ),
        ]


for line in sys.stdin:
    raw_data = json.loads(line)
    content = []
    for step in raw_data["conversations"]:
        content.extend(convert_step(step))

    # Handle finish actions for natural language based tasks
    if (
        isinstance(content[-1], TextObservation) and content[-1].source == "assistant"
    ) or isinstance(content[-1], CodeAction):
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
