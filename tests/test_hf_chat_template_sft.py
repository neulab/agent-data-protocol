import json
from pathlib import Path

import pytest

DATASET_PATH = Path(__file__).parent.parent / "datasets"
HF_TEMPLATE_MODEL = "Qwen/Qwen3-8B"


def get_sample_sft_paths():
    return sorted(DATASET_PATH.glob("*/sample_sft/*.json"))


def record_messages(record):
    return record.get("messages") or record.get("conversations")


@pytest.fixture(scope="session")
def qwen3_tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(
        HF_TEMPLATE_MODEL,
        trust_remote_code=True,
    )


@pytest.mark.parametrize("sample_path", get_sample_sft_paths())
def test_sample_sft_records_apply_qwen3_chat_template(sample_path, qwen3_tokenizer):
    records = json.loads(sample_path.read_text())
    if not records:
        pytest.skip(f"No SFT records in {sample_path}")

    for record_index, record in enumerate(records):
        conversations = record_messages(record)
        assert isinstance(conversations, list), (
            f"{sample_path} record {record_index} must contain a messages or conversations list"
        )
        for message_index, message in enumerate(conversations):
            assert set(message) >= {"role", "content"}, (
                f"{sample_path} record {record_index} message {message_index} must use "
                "Hugging Face role/content fields"
            )
            assert message["role"] in {"system", "user", "assistant", "tool"}, (
                f"{sample_path} record {record_index} message {message_index} has "
                f"unsupported role {message['role']!r}"
            )
            assert isinstance(message["content"], str), (
                f"{sample_path} record {record_index} message {message_index} content "
                "must be a string"
            )

        rendered = qwen3_tokenizer.apply_chat_template(
            conversations,
            tools=record.get("tools"),
            tokenize=False,
            add_generation_prompt=False,
        )
        assert rendered, (
            f"{sample_path} record {record_index} rendered to an empty Qwen3 chat template"
        )
