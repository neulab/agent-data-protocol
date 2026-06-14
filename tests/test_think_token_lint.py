import json
import re
from pathlib import Path

import pytest

DATASET_PATH = Path(__file__).parent.parent / "datasets"
THINK_TOKEN_RE = re.compile(r"</?think\b", re.IGNORECASE)
ASSISTANT_SFT_ROLES = {"assistant", "function_call", "gpt"}


def message_role(message: dict) -> str | None:
    return message.get("role") or message.get("from")


def message_content(message: dict) -> str:
    return message.get("content") or message.get("value") or ""


def get_sample_std_paths():
    return sorted(DATASET_PATH.glob("*/sample_std.json"))


def get_openhands_v0_sample_sft_paths():
    return sorted(DATASET_PATH.glob("*/sample_sft/openhands_v0.json"))


def load_json(path: Path):
    with path.open("r") as file:
        return json.load(file)


@pytest.mark.parametrize("sample_path", get_sample_std_paths())
def test_standardized_atif_agent_messages_do_not_include_think_tokens(sample_path):
    samples = load_json(sample_path)
    for sample_index, sample in enumerate(samples):
        for step_index, step in enumerate(sample.get("steps", [])):
            if step.get("source") != "agent":
                continue

            content = step.get("message") or ""
            if isinstance(content, list):
                content = "\n".join(part.get("text", "") for part in content)
            assert not THINK_TOKEN_RE.search(content), (
                f"ATIF agent message in {sample_path} sample {sample_index} "
                f"step {step_index} contains <think> tags; move the block to "
                "reasoning_content instead"
            )

            reasoning_content = step.get("reasoning_content") or ""
            assert not THINK_TOKEN_RE.search(reasoning_content), (
                f"ATIF reasoning_content in {sample_path} sample {sample_index} "
                f"step {step_index} should contain reasoning text without raw <think> tags"
            )


@pytest.mark.parametrize("sample_path", get_openhands_v0_sample_sft_paths())
def test_openhands_v0_sft_assistant_messages_do_not_include_think_tokens(sample_path):
    samples = load_json(sample_path)
    for sample_index, sample in enumerate(samples):
        for message_index, message in enumerate(sample.get("conversations", [])):
            if message_role(message) not in ASSISTANT_SFT_ROLES:
                continue

            value = message_content(message)
            assert not THINK_TOKEN_RE.search(value), (
                f"SFT assistant message value in {sample_path} sample {sample_index} "
                f"message {message_index} contains <think> tags"
            )
