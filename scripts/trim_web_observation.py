import copy
import re


def total_length(conv: list[dict]) -> int:
    return sum(len(msg["value"]) for msg in conv)


def trim(conversation: list[dict], max_chars: int = 70000) -> list[dict]:
    def is_action(msg: dict) -> bool:
        return msg["from"] in {"gpt", "function_call"}

    def is_observation(msg: dict) -> bool:
        return msg["from"] in {"human", "observation"}

    def extract_steps(conv: list[dict]) -> list[tuple[int, int]]:
        """Return list of (obs_idx, act_idx) pairs."""
        steps = []
        i = 0
        while i < len(conv) - 1:
            if is_observation(conv[i]) and is_action(conv[i + 1]):
                steps.append((i, i + 1))
                i += 2
            else:
                i += 1
        return steps

    def trim_axtree(value: str, overflow: int) -> str:
        trim_notice = "\nTrimming prompt to meet context window limitations\n"
        match = re.search(
            r"(BEGIN accessibility tree ==============)(.*?)(END accessibility tree ==============)",
            value,
            flags=re.DOTALL,
        )
        if not match:
            return value  # Nothing to trim
        pre, tree_content, post = match.groups()
        trimmed_tree = (
            tree_content[: max(0, len(tree_content) - overflow - len(trim_notice))] + trim_notice
        )
        return value.replace(tree_content, trimmed_tree)

    # No trimming needed
    if total_length(conversation) <= max_chars:
        return conversation

    # Try keeping last 3 steps
    # steps = extract_steps(conversation)
    # trimmed = []
    # if len(steps) > 0:
    #     last_n_steps = steps[-3:]
    #     keep_indices = []
    #     for obs_idx, act_idx in last_n_steps:
    #         keep_indices.extend([obs_idx, act_idx])
    #     keep_indices.sort()
    #     trimmed = [conversation[i] for i in keep_indices]

    # if total_length(trimmed) <= max_chars:
    #     return trimmed

    # Still too long, try trimming axtree in the last observation
    # Find last observation
    steps = conversation
    for i in range(len(steps)):
        if total_length(steps) <= max_chars:
            return steps
        elif not i == len(steps) - 2 and is_observation(steps[i]):
            steps[i]["value"] = trim_axtree(steps[i]["value"], 999999999)
        elif i == len(steps) - 2:
            steps[i]["value"] = trim_axtree(steps[i]["value"], total_length(steps) - max_chars)

    return steps


def parse_line(line):
    out = []
    for i in range(len(line["conversations"]) // 2):
        line_copy = copy.deepcopy(line)
        c = line_copy["conversations"][: ((i + 1) * 2)]
        line_copy["id"] = line_copy["id"] + f"-{i}"
        line_copy["conversations"] = trim(c)
        out.append(line_copy)
    # print(len(f"{line}"), len(f"{out[-1]}"), f'{[total_length(l["conversations"]) + len(l["system"]) for l in out]}')
    return out
