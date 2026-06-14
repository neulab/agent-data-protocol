# ruff: noqa: E402

from __future__ import annotations

import json
import re
import sys

from schema.atif import ATIFTrajectory, normalize_atif_trajectory
from scripts.atif_to_std_common import standardize_tools

ANSWER_FORMAT = (
    "If you think you have got the answer to the question, you should print like this:"
    "\n\n<solution> Your solution here </solution>"
)
FIRST_USER_MESSAGE = re.compile(
    r"(You are an assistant.*\n\nNow, my problem is:)\n(.*)",
    re.DOTALL,
)
OS_OUTPUT = re.compile(r"The output of the OS:\n(.*)", re.DOTALL)
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


def normalize_agenttuning_os_trajectory(trajectory: ATIFTrajectory) -> ATIFTrajectory:
    normalized = standardize_tools(normalize_atif_trajectory(trajectory))
    for step in normalized.steps:
        if step.source == "user" and isinstance(step.message, str):
            first_user_match = FIRST_USER_MESSAGE.match(step.message)
            if first_user_match:
                step.message = (
                    first_user_match.group(2).strip().replace("?bash:`", "?")
                    + "\n\n"
                    + ANSWER_FORMAT
                )
        if not step.observation:
            continue
        for result in step.observation.results:
            if not isinstance(result.content, str):
                continue
            os_output_match = OS_OUTPUT.match(result.content)
            if os_output_match:
                result.content = clean_terminal_output(os_output_match.group(1))
    return normalized


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        trajectory = ATIFTrajectory(**json.loads(line))
        normalized = normalize_agenttuning_os_trajectory(trajectory)
        print(normalized.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main()
