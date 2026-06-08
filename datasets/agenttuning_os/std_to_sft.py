# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

DATASET_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATASET_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DATASET_DIR))

from agents.openhands_sdk import std_to_sft as openhands_sdk_std_to_sft
from raw_to_standardized import clean_terminal_output
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory
from scripts.atif_input import load_trajectory

ANSWER_FORMAT = (
    "If you think you have got the answer to the question, you should print like this:"
    "\n\n<solution> Your solution here </solution>"
)
FIRST_USER_MESSAGE = re.compile(
    r"(You are an assistant.*\n\nNow, my problem is:)\n(.*)",
    re.DOTALL,
)
OS_OUTPUT = re.compile(r"The output of the OS:\n(.*)", re.DOTALL)


def prepare_trajectory_for_sft(trajectory: Trajectory) -> Trajectory:
    prepared = trajectory.model_copy(deep=True)
    for event in prepared.content:
        if not isinstance(event, TextObservation):
            continue
        if event.source == "user":
            first_user_match = FIRST_USER_MESSAGE.match(event.content)
            if first_user_match:
                event.content = (
                    first_user_match.group(2).strip().replace("?bash:`", "?")
                    + "\n\n"
                    + ANSWER_FORMAT
                )
        elif event.source == "environment":
            os_output_match = OS_OUTPUT.match(event.content)
            if os_output_match:
                event.content = clean_terminal_output(os_output_match.group(1))
    return prepared


def process_row(line: str, model: str, dataset_name: str | None = None) -> dict[str, Any]:
    dataset_name = dataset_name or os.getenv("MY_DATASET") or "agenttuning_os"
    trajectory = prepare_trajectory_for_sft(load_trajectory(line))
    return openhands_sdk_std_to_sft.process_trajectory(trajectory, model, dataset_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert AgentTuning OS ATIF std data to OpenHands SDK SFT format"
    )
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    args = parser.parse_args()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        print(json.dumps(process_row(line, args.model), ensure_ascii=False))


if __name__ == "__main__":
    main()
