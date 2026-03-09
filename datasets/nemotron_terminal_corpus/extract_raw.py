import json

from datasets import load_dataset


def normalize_conversation(conv):
    """Normalize a conversation turn to have consistent role naming."""
    return {"role": conv["role"], "content": conv["content"]}


def filter_valid_trajectory(sample):
    """Filter trajectories to ensure they have valid conversation structure."""
    conversations = sample["conversations"]

    # Must have at least one user and one assistant turn
    has_user = any(turn["role"] == "user" for turn in conversations)
    has_assistant = any(turn["role"] == "assistant" for turn in conversations)

    if not (has_user and has_assistant):
        return False

    # First turn should not be assistant
    if conversations[0]["role"] == "assistant":
        return False

    return True


# Load dataset from skill_based_easy config (most accessible)
# Other configs: skill_based_medium, skill_based_mixed
dataset = load_dataset("nvidia/Nemotron-Terminal-Corpus", "skill_based_easy", split="train")

for sample in dataset:
    if not filter_valid_trajectory(sample):
        continue

    output = {
        "conversations": [normalize_conversation(turn) for turn in sample["conversations"]],
        "agent": sample.get("agent"),
        "model": sample.get("model"),
        "model_provider": sample.get("model_provider"),
        "date": sample.get("date"),
        "task": sample.get("task"),
        "episode": sample.get("episode"),
        "run_id": sample.get("run_id"),
        "trial_name": sample.get("trial_name"),
        "enable_thinking": sample.get("enable_thinking"),
    }
    print(json.dumps(output))
