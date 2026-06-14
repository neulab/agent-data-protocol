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
SAMPLE_IDS = {
    "alfworld": ["alfworld_155", "alfworld_219", "alfworld_56", "alfworld_58", "alfworld_149"],
    "db": ["db_100", "db_356", "db_493", "db_516", "db_531"],
    "kg": ["kg_132", "kg_260", "kg_164", "kg_221", "kg_12"],
    "webshop": ["webshop_330", "webshop_190", "webshop_243", "webshop_89", "webshop_26"],
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
        sample_ids = SAMPLE_IDS.get(config_name)
        if sample_ids is not None:
            selected = {}
            sample_id_set = set(sample_ids)
            for sample in dataset:
                if sample.get("id") in sample_id_set and has_valid_conversation(sample):
                    selected[sample["id"]] = json_safe(sample)
            for sample_id in sample_ids:
                if sample_id in selected:
                    print(json.dumps(selected[sample_id], ensure_ascii=False))
            return
        for sample in dataset:
            if not has_valid_conversation(sample):
                continue
            print(json.dumps(json_safe(sample), ensure_ascii=False))
    except BrokenPipeError:
        sys.stdout = open(os.devnull, "w")
        sys.exit(0)


if __name__ == "__main__":
    main("webshop")
