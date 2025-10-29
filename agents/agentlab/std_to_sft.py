import json
import os
import re
import sys

from agents.openhands.std_to_sft import main_with_args as main_openhands
from schema.trajectory import Trajectory

dataset = os.getenv("MY_DATASET")
assert dataset, "Please set the environment variable MY_DATASET"

with open("agents/agentlab/system.txt") as f:
    system = f.read()
with open("agents/agentlab/action_space.txt") as f:
    action_space = f.read()
with open("agents/agentlab/suffix.txt") as f:
    suffix = f.read()


def process_row(line):
    std_dataset = [json.loads(line)]
    std_data = std_dataset[0]
    trajectory = Trajectory(**std_data)
    events = trajectory.content
    output_line = json.loads(main_openhands(line, is_web=True, api_env="browser"))
    goal = "# Goal\n" + events[0].content + "\n\n"
    past_actions = []
    observation = ""
    for step in range(len(output_line["conversations"])):
        if step % 2 == 1:
            match = re.search(
                r"(<function=browser>\n<parameter=code>\n)(.*?)(\n</parameter>\n</function>)",
                output_line["conversations"][step]["value"],
                flags=re.DOTALL,
            )
            if not match:
                action = ""
            else:
                thought, action, _ = match.groups()
                action = json.dumps({"thought": thought, "action": action}).strip()
                past_actions.append(action)
                action = {"role": "assistant", "content": action}
                if observation:
                    return {
                        "id": f"{trajectory.id}-{step // 2}",
                        "conversations": [observation, action],
                        "system": system,
                    }
        else:
            match = re.search(
                r"(BEGIN accessibility tree ==============)(.*?)(END accessibility tree ==============)",
                output_line["conversations"][step]["value"],
                flags=re.DOTALL,
            )
            if not match:
                tree = ""
            else:
                _, tree, _ = match.groups()
                tree = "# Current page Accessibility Tree\n" + tree + "\n\n"
            history = "# History of past actions\n" + "\n".join(past_actions) + "\n\n"
            observation = {"role": "user", "content": goal + tree + action_space + history + suffix}


def main():
    for line in sys.stdin:
        output_line = process_row(line)
        if output_line:
            output_line = json.dumps(output_line)
            print(output_line)


if __name__ == "__main__":
    main()
