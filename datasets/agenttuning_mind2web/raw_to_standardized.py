import json
import re
import sys
from typing import Tuple

from schema.action.action import Action
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


def parse_thought_and_answer(message: str) -> Tuple[str, str]:
    """
    Extracts the thought and answer
    """
    match = re.search(r"Thought:\s*(.*?)\s*Answer:\s*(.*)", message, re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse Thought and Answer from {message}.")

    thought = match.group(1).strip()
    answer = match.group(2).strip()
    return thought, answer


def convert_step(step: dict[str, str]) -> list[Action | Observation]:
    if step["role"] == "user":
        return [
            TextObservation(content=step["content"], source=step["role"]),
        ]
    else:
        assert step["role"] == "assistant"
        thought, answer = parse_thought_and_answer(step["content"])

        return [MessageAction(content=f"<finish> {answer} </finish>", description=thought)]


for line in sys.stdin:
    raw_data = json.loads(line)
    assert len(raw_data["conversations"]) == 2
    content = []

    try:
        for step in raw_data["conversations"]:
            content.extend(convert_step(step))
    except:
        continue

    assert len(content) == 2
    match = re.search(r"\b([A-Z])\.", content[1].content)
    if not match:
        raise ValueError(f"No valid option key found in: {content[1].content}")
    option_key = match.group(1)
    # All answers should contain an option key in the user message
    if f"{option_key}." not in content[0].content:
        continue

    # Standardize the data
    standardize_data = Trajectory(
        id=raw_data["id"],
        content=content,
    )

    # Print the standardized data
    print(standardize_data.model_dump_json())
