import json
import re
import sys
from typing import Tuple

from schema.action.action import Action
from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.tool_call_links import create_trajectory_with_tool_call_links


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


def selected_choice_markup(user_message: str, option_key: str) -> str:
    match = re.search(
        rf"^\s*{re.escape(option_key)}\.\s*(.*?)\s*$",
        user_message,
        re.MULTILINE,
    )
    return match.group(1).strip() if match else ""


def element_index(choice_markup: str, option_key: str) -> int | str:
    match = re.search(r"\bid=(\d+)", choice_markup)
    if match:
        return int(match.group(1))
    return option_key


def parse_mind2web_action(
    thought: str, answer: str, user_message: str
) -> list[Action | Observation] | None:
    match = re.match(
        r"(?P<option>[A-Z])\.\s*Action:\s*(?P<action>[A-Z]+)"
        r"(?:\s*Value:\s*(?P<value>.*))?\s*$",
        answer,
        re.DOTALL,
    )
    if match is None:
        return None

    option_key = match.group("option")
    action = match.group("action").upper()
    value = (match.group("value") or "").strip()
    choice = selected_choice_markup(user_message, option_key)
    index = element_index(choice, option_key)

    if action == "CLICK":
        api_action = ApiAction(
            function="click",
            kwargs={"bid": index},
            description=thought,
        )
    elif action in {"TYPE", "SET"}:
        api_action = ApiAction(
            function="fill",
            kwargs={"bid": index, "value": value},
            description=thought,
        )
    else:
        return None

    result = TextObservation(
        content=(
            f"Recorded Mind2Web action prediction: option {option_key}, "
            f"action {action}" + (f", value {value}." if value else ".")
        ),
        source="environment",
    )
    finish = MessageAction(content=f"<finish> {answer} </finish>", description=thought)
    return [api_action, result, finish]


def convert_step(
    step: dict[str, str], previous_user: str | None = None
) -> list[Action | Observation]:
    if step["role"] == "user":
        return [
            TextObservation(content=step["content"], source=step["role"]),
        ]
    else:
        assert step["role"] == "assistant"
        thought, answer = parse_thought_and_answer(step["content"])
        if previous_user is not None:
            action_events = parse_mind2web_action(thought, answer, previous_user)
            if action_events is not None:
                return action_events

        return [MessageAction(content=f"<finish> {answer} </finish>", description=thought)]


for line in sys.stdin:
    raw_data = json.loads(line)
    assert len(raw_data["conversations"]) == 2
    content = []

    try:
        previous_user = None
        for step in raw_data["conversations"]:
            content.extend(convert_step(step, previous_user))
            if step["role"] == "user":
                previous_user = step["content"]
    except:
        continue

    final_message = next(item for item in reversed(content) if isinstance(item, MessageAction))
    match = re.search(r"\b([A-Z])\.", final_message.content)
    if not match:
        raise ValueError(f"No valid option key found in: {final_message.content}")
    option_key = match.group(1)
    # All answers should contain an option key in the user message
    if f"{option_key}." not in content[0].content:
        continue

    # Standardize the data
    standardize_data = create_trajectory_with_tool_call_links(
        id=raw_data["id"],
        content=content,
        details={
            "source": "THUDM/AgentInstruct",
            "source_split": "mind2web",
            "source_id": raw_data["id"],
        },
    )

    # Print the standardized data
    print(
        standardize_data.model_dump_json(
            # Suppress optional schema fields unused by this dataset.
            exclude={"content": {"__all__": {"reasoning_content", "reward"}}}
        )
    )
