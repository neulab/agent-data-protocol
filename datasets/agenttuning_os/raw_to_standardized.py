import json
import re
import sys

from schema.action.action import Action
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


def convert_first_user_message(first_user_message_regex: re.Match[str]) -> list[Observation]:
    """
    Extracts and formats the essential parts of a system prompt.
    """
    assert "You are an assistant" in first_user_message_regex.group(1)
    # assert re.search(r"\"bash\"", first_user_message_regex.group(1), re.DOTALL)
    # assert re.search(r"Act: finish", first_user_message_regex.group(1), re.DOTALL)
    # assert re.search(r"```bash\n(.*?)\n```", first_user_message_regex.group(1), re.DOTALL)
    # assert re.search(r"answer(.*)", first_user_message_regex.group(1), re.DOTALL)

    system_msg = """If you think you have got the answer to the question, you should print like this:\n\n<solution> Your solution here </solution>"""
    return [
        TextObservation(
            content=first_user_message_regex.group(2).strip().replace("?bash:`", "?")
            + "\n\n"
            + system_msg,
            source="user",
        )
    ]


def convert_step(step: dict[str, str]) -> list[Action | Observation]:
    # parse first user message
    first_user_message_regex = re.match(
        r"(You are an assistant.*\n\nNow, my problem is:)\n(.*)",  # noqa
        step["content"],
        re.DOTALL,
    )
    if first_user_message_regex:
        return convert_first_user_message(first_user_message_regex)

    code_act_regex = re.match(r"Think: (.*)\n\nAct: (.*)", step["content"], re.DOTALL)
    code_obs_regex = re.match(r"The output of the OS:\n(.*)", step["content"], re.DOTALL)

    if code_act_regex:
        bash_extract_regex = re.match(
            r"bash\n\n```bash\n(.*)\n```|bash \n\n```bash\n(.*)\n```|bash\n  \n```bash\n(.*)\n```",
            code_act_regex.group(2),
            re.DOTALL,
        )
        answer_extract_regex = re.match(r"answer\((.*)\)", code_act_regex.group(2), re.DOTALL)
        finish_extract_regex = re.match(r"finish", code_act_regex.group(2), re.DOTALL)
        if bash_extract_regex:
            return [
                CodeAction(
                    language="bash",
                    content=bash_extract_regex.group(1)
                    or bash_extract_regex.group(2)
                    or bash_extract_regex.group(3),
                    description=code_act_regex.group(1),
                ),
            ]
        elif answer_extract_regex:
            return [
                MessageAction(
                    content=f"<solution> {answer_extract_regex.group(1)} </solution>",
                    description=code_act_regex.group(1),
                ),
            ]
        elif finish_extract_regex:
            return [
                MessageAction(
                    content="<finish></finish>",
                    description=code_act_regex.group(1),
                ),
            ]
        else:
            raise ValueError(
                f"Could not extract code from code action in {json.dumps(step, indent=2)}"
            )

    elif code_obs_regex:
        return [
            TextObservation(content=code_obs_regex.group(1), source="os"),
        ]

    else:
        return [
            TextObservation(
                content=step["content"]
                .replace("Thought:", "THOUGHT:")
                .replace("Action:", "ACTION:")
                .replace("Observation:", "OBSERVATION:"),
                source=step["role"] if step["role"] != "system" else "user",
            ),
        ]


for line in sys.stdin:
    raw_data = json.loads(line)
    content = []
    for step in raw_data["conversations"]:
        content.extend(convert_step(step))

    # Handle finish actions
    if isinstance(content[-1], MessageAction) and "<solution>" in content[-1].content:
        content[-1].content = f"<finish> {content[-1].content} </finish>"

    # Standardize the data
    standardize_data = Trajectory(
        id=raw_data["id"],
        content=content,
    )

    # Print the standardized data
    print(standardize_data.model_dump_json())
