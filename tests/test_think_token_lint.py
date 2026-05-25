import json
import re
from pathlib import Path

import pytest

DATASET_PATH = Path(__file__).parent.parent / "datasets"
THINK_TOKEN_RE = re.compile(r"</?think\b", re.IGNORECASE)
ASSISTANT_SFT_ROLES = {"assistant", "function_call", "gpt"}


def get_sample_std_paths():
    return sorted(DATASET_PATH.glob("*/sample_std.json"))


def get_openhands_v0_sample_sft_paths():
    return sorted(DATASET_PATH.glob("*/sample_sft/openhands_v0.json"))


def load_json(path: Path):
    with path.open("r") as file:
        return json.load(file)


@pytest.mark.parametrize("sample_path", get_sample_std_paths())
def test_standardized_message_actions_do_not_include_think_tokens(sample_path):
    samples = load_json(sample_path)
    for sample_index, sample in enumerate(samples):
        for event_index, event in enumerate(sample.get("content", [])):
            if event.get("class_") != "message_action":
                continue

            content = event.get("content") or ""
            assert not THINK_TOKEN_RE.search(content), (
                f"MessageAction.content in {sample_path} sample {sample_index} "
                f"event {event_index} contains <think> tags; move the block to "
                "reasoning_content instead"
            )

            reasoning_content = event.get("reasoning_content") or ""
            assert not THINK_TOKEN_RE.search(reasoning_content), (
                f"MessageAction.reasoning_content in {sample_path} sample {sample_index} "
                f"event {event_index} should contain reasoning text without raw <think> tags"
            )


@pytest.mark.parametrize("sample_path", get_openhands_v0_sample_sft_paths())
def test_openhands_v0_sft_assistant_messages_do_not_include_think_tokens(sample_path):
    samples = load_json(sample_path)
    for sample_index, sample in enumerate(samples):
        for message_index, message in enumerate(sample.get("conversations", [])):
            if message.get("from") not in ASSISTANT_SFT_ROLES:
                continue

            value = message.get("value") or ""
            assert not THINK_TOKEN_RE.search(value), (
                f"SFT assistant message value in {sample_path} sample {sample_index} "
                f"message {message_index} contains <think> tags"
            )
