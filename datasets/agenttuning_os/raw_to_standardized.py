import json
import re
import sys

from schema.action.action import Action
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.tool_call_links import create_trajectory_with_tool_call_links
from schema.trajectory import Trajectory

OSC_SEQUENCE = re.compile(r"\x1b\][^\x07]*(?:\x07|\x1b\\)")
ANSI_SEQUENCE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SHELL_PROMPT = re.compile(r"(?:\r?\n)?root@[A-Za-z0-9_.-]+:[^\r\n#]*# ?$")


def clean_terminal_output(content: str) -> str:
    content = OSC_SEQUENCE.sub("", content)
    content = ANSI_SEQUENCE.sub("", content)
    content = content.replace("\x07", "")
    content = SHELL_PROMPT.sub("", content)
    content = content.replace("\r\n", "\n").replace("\r", "")
    return content.strip()


def convert_first_user_message(first_user_message_regex: re.Match[str]) -> list[Observation]:
    """
    Extracts the user task and drops the source system prompt.
    """
    assert "You are an assistant" in first_user_message_regex.group(1)
    answer_format = (
        "If you think you have got the answer to the question, you should print like this:"
        "\n\n<solution> Your solution here </solution>"
    )
    return [
        TextObservation(
            content=(
                first_user_message_regex.group(2).strip().replace("?bash:`", "?")
                + "\n\n"
                + answer_format
            ),
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
            TextObservation(
                content=clean_terminal_output(code_obs_regex.group(1)),
                source="environment",
            ),
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


def process_raw_data(raw_data: dict) -> Trajectory:
    content = []
    for step in raw_data["conversations"]:
        content.extend(convert_step(step))

    # Handle finish actions
    if isinstance(content[-1], MessageAction) and "<solution>" in content[-1].content:
        content[-1].content = f"<finish> {content[-1].content} </finish>"

    return create_trajectory_with_tool_call_links(
        id=raw_data["id"],
        content=content,
    )


def main() -> None:
    for line in sys.stdin:
        raw_data = json.loads(line)
        standardize_data = process_raw_data(raw_data)
        print(
            standardize_data.model_dump_json(
                exclude={"content": {"__all__": {"reasoning_content", "reward"}}}
            )
        )


if __name__ == "__main__":
    main()
