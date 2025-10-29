import json
import re

with open("agents/agentlab/system.txt") as f:
    system = f.read()
with open("agents/agentlab/action_space.txt") as f:
    action_space = f.read()
with open("agents/agentlab/suffix.txt") as f:
    suffix = f.read()

with open("full_sft.jsonl") as f:
    f = f.readlines()
sft = [json.loads(line) for line in f]
id_to_sft = {line["id"]: line["conversations"] for line in sft}
with open("full_std.jsonl") as f:
    f = f.readlines()
std = [json.loads(line) for line in f]
id_to_std = {line["id"]: line["content"] for line in std}

output = []
for id in id_to_sft:
    # if not len(id_to_sft[id]) == len(id_to_std[id]): continue
    goal = "# Goal\n" + id_to_std[id][0]["content"] + "\n\n"
    past_actions = []
    observation = ""
    for step in range(len(id_to_sft[id])):
        if step % 2 == 1:
            match = re.search(
                r"(<function=browser>\n<parameter=code>\n)(.*?)(\n</parameter>\n</function>)",
                id_to_sft[id][step]["value"],
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
                    output.append(
                        {
                            "id": f"{id}-{step // 2}",
                            "conversations": [observation, action],
                            "system": system,
                        }
                    )
        else:
            match = re.search(
                r"(BEGIN accessibility tree ==============)(.*?)(END accessibility tree ==============)",
                id_to_sft[id][step]["value"],
                flags=re.DOTALL,
            )
            if not match:
                tree = ""
            else:
                _, tree, _ = match.groups()
                tree = "# Current page Accessibility Tree\n" + tree + "\n\n"
            history = "# History of past actions\n" + "\n".join(past_actions) + "\n\n"
            observation = {"role": "user", "content": goal + tree + action_space + history + suffix}

with open("synatra.json", "w") as f:
    json.dump(output, f, indent=2)
with open("synatra-sample.json", "w") as f:
    json.dump(output[:5], f, indent=2)
