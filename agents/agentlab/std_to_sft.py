import json
import os
import re
import sys

from agents.openhands_v0.std_to_sft import main_with_args as main_openhands_v0
from scripts.atif_input import load_trajectory

dataset = os.getenv("MY_DATASET")
assert dataset, "Please set the environment variable MY_DATASET"

with open("agents/agentlab/action_space.txt") as f:
    action_space = f.read().strip()
with open("agents/agentlab/suffix.txt") as f:
    suffix = f.read().strip()

system = (
    "# Instructions\n"
    "You are a UI Assistant, your goal is to help the user perform tasks using a web browser.\n"
    "Review the instructions from the user, the current state of the page and all other "
    "information to find the best possible next action to accomplish your goal. Your answer "
    "will be interpreted and executed by a program, make sure to follow the formatting "
    "instructions.\n\n"
    f"{action_space}\n\n{suffix}"
)


def process_row(line):
    trajectory = load_trajectory(line)
    events = trajectory.content
    output_line = json.loads(main_openhands_v0(line, is_web=True, api_env="browser"))
    goal = "# Goal\n" + events[0].content + "\n\n"
    past_actions = []
    observation = ""
    ret = []
    for step in range(len(output_line["conversations"])):
        message_content = output_line["conversations"][step]["content"]
        if step % 2 == 1:
            match = re.search(
                # r"(<function=browser>\n<parameter=code>\n)(.*?)(\n</parameter>\n</function>)",
                r"^(?P<thought>.*?)<function=browser>\s*<parameter=code>\s*(?P<action>.*?)\s*</parameter>",
                message_content,
                flags=re.DOTALL,
            )
            if not match:
                match = re.search(
                    r"^(?P<thought>.*?)<function=finish>\s*<parameter=message>\s*(?P<action>.*?)\s*</parameter>",
                    message_content,
                    flags=re.DOTALL,
                )
                if match:
                    thought = match.group("thought").strip()
                    message = match.group("action").strip()
                elif "<function=" not in message_content:
                    thought = ""
                    message = message_content
                else:
                    raise ValueError(f"Unsupported AgentLab function call: {message_content}")
                action = f'send_msg_to_user(text="{message}")'

            else:
                thought = match.group("thought").strip()
                action = match.group("action").strip()
            action = json.dumps({"thought": thought, "action": action}).strip()
            past_actions.append(action)
            action = {"role": "assistant", "content": action}
            if observation:
                ret.append(
                    {
                        "id": f"{trajectory.id}-{step // 2}",
                        "conversations": [observation, action],
                        "system": system,
                    }
                )
            else:
                raise ValueError(f"no observation: {message_content}")
        else:
            match = re.search(
                r"(============== BEGIN accessibility tree ==============)(.*?)(============== END accessibility tree ==============)",
                message_content,
                flags=re.DOTALL,
            )
            if not match:
                tree = ""
            else:
                _, tree, _ = match.groups()
                tree = "# Current page Accessibility Tree\n" + tree.strip() + "\n\n"
            history = "\n\n\n\n\n# History of past actions\n" + "\n".join(past_actions) + "\n\n"
            observation = {"role": "user", "content": goal + tree + history}
    return ret


def main():
    for line in sys.stdin:
        output_lines = process_row(line)
        if output_lines:
            for output_line in output_lines:
                output_line = json.dumps(output_line)
                print(output_line)


if __name__ == "__main__":
    main()
