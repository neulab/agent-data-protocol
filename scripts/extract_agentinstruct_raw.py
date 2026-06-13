from __future__ import annotations

import json
import os
import sys
from typing import Any

from datasets import load_dataset


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def main(config_name: str) -> None:
    try:
        dataset = load_dataset("THUDM/AgentInstruct")[config_name]
        for sample in dataset:
            print(json.dumps(json_safe(sample), ensure_ascii=False))
    except BrokenPipeError:
        sys.stdout = open(os.devnull, "w")
        sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m scripts.extract_agentinstruct_raw <config_name>")
    main(sys.argv[1])
