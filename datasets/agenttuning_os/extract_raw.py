from __future__ import annotations

import json
import os
import sys
from typing import Any

from datasets import load_dataset

ROLE_MAP = {
    "user": "user",
    "human": "user",
    "chatgpt": "assistant",
    "gpt": "assistant",
    "bard": "assistant",
    "assistant": "assistant",
    "system": "system",
}

def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def role(turn: dict[str, Any]) -> str | None:
    raw_role = turn.get("role") or turn.get("from")
    return ROLE_MAP.get(str(raw_role)) if raw_role is not None else None


def has_valid_conversation(sample: dict[str, Any]) -> bool:
    conversations = sample.get("conversations")
    if not isinstance(conversations, list):
        return False
    roles = [role(turn) for turn in conversations if isinstance(turn, dict)]
    if "user" not in roles or "assistant" not in roles:
        return False
    if roles and roles[0] == "assistant":
        return False
    return not any(
        roles[index] == "assistant" and roles[index + 1] == "assistant"
        for index in range(len(roles) - 1)
    )


def main(config_name: str) -> None:
    try:
        dataset = load_dataset("THUDM/AgentInstruct")[config_name]
        for sample in dataset:
            if not has_valid_conversation(sample):
                continue
            print(json.dumps(json_safe(sample), ensure_ascii=False))
    except BrokenPipeError:
        sys.stdout = open(os.devnull, "w")
        sys.exit(0)


if __name__ == "__main__":
    main("os")
