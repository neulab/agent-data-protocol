import json
import re
import sys

from schema.action.action import Action
from schema.action.code import CodeAction
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
        TextObservation(content=system_prompt, source="environment"),
        TextObservation(content="Ok? Understood?", source="user"),
    ]


def format_code(raw_code: str) -> str:
    # Format the function name and argument list and add quotes
    match = re.match(r"(\w+)\((.*)\)", raw_code)
    if not match:
        raise ValueError(f"Invalid function call format: {raw_code}")
    func_name = match.group(1)
    args_str = match.group(2)
    if not args_str:
        return f"{func_name}()"
    args = [arg.strip() for arg in args_str.split(",")]
    quoted_args = [f"'{arg}'" for arg in args]
    return f"{func_name}({', '.join(quoted_args)})"


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
                content=f"<finish> <solution> {answer_extract_regex.group(2)} </solution> </finish>",
                description=answer_extract_regex.group(1).strip(),
            )
        ]

    # parse assistant message
    elif step["role"] == "assistant":
        match = re.search(r"Thought:\s*(.*?)\nAction:\s*(.*)", step["content"], re.DOTALL)
        if match:
            thought = match.group(1).strip()
            action = match.group(2).strip()
            return [
                CodeAction(
                    language="python",
                    content=format_code(action),
                    description=thought,
                ),
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
    if not isinstance(content[-1], MessageAction) or "<finish>" not in content[-1].content:
        continue

    # Standardize the data
    standardize_data = Trajectory(
        id=raw_data["id"],
        content=content,
    )

    # Print the standardized data
    print(standardize_data.model_dump_json())
