import json
import os
import re
import sys

from agents.openhands.std_to_sft import main_with_args as main_openhands
from schema.trajectory import Trajectory

dataset = os.getenv("MY_DATASET")
assert dataset, "Please set the environment variable MY_DATASET"

system = "# Instructions\nYou are a UI Assistant, your goal is to help the user perform tasks using a web browser. \nReview the instructions from the user, the current state of the page and all other information to find the best possible next action to accomplish your goal. Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.\n"
with open("agents/agentlab/action_space.txt") as f:
    action_space = f.read().strip()
with open("agents/agentlab/suffix.txt") as f:
    suffix = f.read().strip()


def process_row(line):
    std_dataset = [json.loads(line)]
    std_data = std_dataset[0]
    trajectory = Trajectory(**std_data)
    events = trajectory.content
    output_line = json.loads(main_openhands(line, is_web=True, api_env="browser"))
    goal = "# Goal\n" + events[0].content + "\n\n"
    past_actions = []
    observation = ""
    ret = []
    for step in range(len(output_line["conversations"])):
        if step % 2 == 1:
            match = re.search(
                # r"(<function=browser>\n<parameter=code>\n)(.*?)(\n</parameter>\n</function>)",
                r"^(?P<thought>.*?)<function=browser>\s*<parameter=code>\s*(?P<action>.*?)\s*</parameter>",
                output_line["conversations"][step]["value"],
                flags=re.DOTALL,
            )
            if not match:
                match = re.search(
                    r"^(?P<thought>.*?)<function=finish>\s*<parameter=message>\s*(?P<action>.*?)\s*</parameter>",
                    output_line["conversations"][step]["value"],
                    flags=re.DOTALL,
                )
                if match:
                    thought = match.group("thought").strip()
                    message = match.group("action").strip()
                elif "<function=" not in output_line["conversations"][step]["value"]:
                    thought = ""
                    message = output_line["conversations"][step]["value"]
                else:
                    raise ValueError(f"no match: {output_line['conversations'][step]['value']}")
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
                raise ValueError(f"no observation: {output_line['conversations'][step]['value']}")
        else:
            match = re.search(
                r"(============== BEGIN accessibility tree ==============)(.*?)(============== END accessibility tree ==============)",
                output_line["conversations"][step]["value"],
                flags=re.DOTALL,
            )
            if not match:
                tree = ""
            else:
                _, tree, _ = match.groups()
                tree = "# Current page Accessibility Tree\n" + tree.strip() + "\n\n"
            history = "\n\n\n\n\n# History of past actions\n" + "\n".join(past_actions) + "\n\n"
            observation = {"role": "user", "content": goal + tree + action_space + history + suffix}
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
