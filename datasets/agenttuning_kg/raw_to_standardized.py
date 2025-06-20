import inspect
import json
import os
import random
import re
import sys
import types
from typing import Tuple

from schema.action.action import Action
from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


def convert_system(system_regex: re.Match[str]) -> list[Observation]:
    """
    Extracts and formats the essential parts of a system prompt,
    removing tool descriptions since we already have these in the api.py.
    """
    assert re.search(r"Final Answer: #3", system_regex.group(1), re.DOTALL)

    system_prompt = (
        system_regex.group(1)
        .replace(r"Final Answer: #3", r"<solution> #3 </solution>")
        .replace(r"Final Answer: #id", r"<solution> #id </solution>")
    )
    system_prompt += system_regex.group(2)
    system_prompt = system_prompt.split("\n\n")
    system_prompt = system_prompt[0] + "\n\n" + system_prompt[-1]

    return [
        TextObservation(content=system_prompt + "\n\n" + "Ok? Understood?", source="user"),
    ]


# Extracts function signatures and docstrings from a python file content string
def get_api_sigs() -> dict[str, list]:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "api.py")) as f:
        api_content = f.read()
    api_module = types.ModuleType("api_module")
    exec(api_content, api_module.__dict__)
    functions = inspect.getmembers(api_module, inspect.isfunction)
    sigs = {}
    for name, func in functions:
        sig = inspect.signature(func)
        sigs[name] = []
        for arg_name, _ in sig.parameters.items():
            sigs[name].append(arg_name)
    return sigs


SIGS = get_api_sigs()


# Format the function name and argument list and add quotes
def format_code(raw_code: str) -> Tuple[str, dict]:
    match = re.match(r"(\w+)\((.*)\)", raw_code)
    if not match:
        raise ValueError(f"Invalid function call format: {raw_code}")
    func_name = match.group(1)
    args_str = match.group(2)
    if func_name not in SIGS:
        raise ValueError(f"Invalid function call: {raw_code}")
    if not args_str:
        return func_name, {}
    args = [arg.strip() for arg in args_str.split(",")]
    if not len(args) == len(SIGS[func_name]):
        raise ValueError(f"Invalid function call: {raw_code}")
    args = {SIGS[func_name][i]: f"'{arg}'" for i, arg in enumerate(args)}
    return func_name, args


def convert_step(step: dict[str, str]) -> list[Action | Observation]:
    # parse system prompt
    system_regex = re.match(
        r"(You are an agent that answers questions.*)\n(.*)",  # noqa
        step["content"],
        re.DOTALL,
    )
    if system_regex:
        return convert_system(system_regex)

    # parse final answer
    elif step["role"] == "assistant" and "Final Answer:" in step["content"]:
        answer_extract_regex = re.search(r"(.*)Final Answer:\s*(.*)", step["content"], re.DOTALL)
        return [
            MessageAction(
                content=f"<solution> {answer_extract_regex.group(2)} </solution>",
                description=answer_extract_regex.group(1).strip().replace("Thought: ", ""),
            )
        ]

    # parse assistant message
    elif step["role"] == "assistant":
        match = re.search(r"Thought:\s*(.*?)\nAction:\s*(.*)", step["content"], re.DOTALL)
        if match:
            thought = match.group(1).strip()
            action = match.group(2).strip()
            api_name, kwargs = format_code(action)

            return [
                ApiAction(
                    function=api_name,
                    kwargs=kwargs,
                    description=thought,
                )
            ]
        else:
            return [TextObservation(content=step["content"], source="agent")]

    # parse user message
    else:
        return [
            TextObservation(
                content=step["content"].replace("Observation:", ""),
                source=step["role"] if step["role"] != "system" else "environment",
            ),
        ]


for line in sys.stdin:
    raw_data = json.loads(line)

    content = []
    try:
        for step in raw_data["conversations"]:
            content.extend(convert_step(step))
    except:
        continue

    # All correctly parsed instances should have a finish message
    if not isinstance(content[-1], MessageAction) or "<solution>" not in content[-1].content:
        continue

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

    # Standardize the data
    standardize_data = Trajectory(
        id=raw_data["id"],
        content=content,
    )

    # Print the standardized data
    print(standardize_data.model_dump_json())
