import importlib.util
import json
from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent / "datasets" / "agenttuning_os"


def load_raw_to_standardized():
    converter_path = DATASET_DIR / "raw_to_standardized.py"
    spec = importlib.util.spec_from_file_location("agenttuning_os_converter", converter_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iter_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from iter_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_strings(item)


def test_clean_terminal_output_removes_control_sequences_and_shell_prompt():
    converter = load_raw_to_standardized()

    assert (
        converter.clean_terminal_output(
            "\n262\r\n\x1b]0;root@287e51facd12: /\x07root@287e51facd12:/# \x1b[K"
        )
        == "262"
    )
    assert (
        converter.clean_terminal_output(
            "\n\x1b[01;31mroot\x1b[m\x1b[K 1 bash\r\n\x1b]0;root@16f05fb4ec8c: /\x07root@16f05fb4ec8c:/#"
        )
        == "root 1 bash"
    )
    assert (
        converter.clean_terminal_output(
            "\n\x1b]0;root@287e51facd12: /\x07root@287e51facd12:/# \x1b[K"
        )
        == ""
    )
    assert converter.clean_terminal_output("first\r\nsecond\rthird") == "first\nsecondthird"


def test_agenttuning_os_derived_samples_do_not_include_terminal_control_sequences():
    sample_paths = [
        DATASET_DIR / "sample_std.json",
        DATASET_DIR / "sample_sft.json",
        *sorted((DATASET_DIR / "sample_sft").glob("*.json")),
    ]

    for sample_path in sample_paths:
        sample = json.loads(sample_path.read_text())
        noisy_values = [
            value
            for value in iter_strings(sample)
            if "\x1b" in value or "\x07" in value or "\r" in value
        ]
        assert not noisy_values, f"{sample_path} contains terminal artifacts"


def test_agenttuning_os_derived_samples_do_not_include_source_system_prompt():
    sample_paths = [
        DATASET_DIR / "sample_std.json",
        DATASET_DIR / "sample_sft.json",
        *sorted((DATASET_DIR / "sample_sft").glob("*.json")),
    ]
    source_prompt_fragments = [
        "You are an assistant that will act like a person",
        "take exact one of the three actions",
        "Act: bash",
    ]

    for sample_path in sample_paths:
        sample = json.loads(sample_path.read_text())
        leaked_values = [
            value
            for value in iter_strings(sample)
            if any(fragment in value for fragment in source_prompt_fragments)
        ]
        assert not leaked_values, f"{sample_path} contains source system prompt text"


def test_agenttuning_os_samples_keep_solution_format_instruction():
    sample_paths = [
        DATASET_DIR / "sample_std.json",
        DATASET_DIR / "sample_sft.json",
        DATASET_DIR / "sample_sft" / "sample_sft_openhands.json",
        DATASET_DIR / "sample_sft" / "sample_sft_sweagent.json",
    ]
    required_fragment = "<solution> Your solution here </solution>"

    for sample_path in sample_paths:
        sample = json.loads(sample_path.read_text())
        values = list(iter_strings(sample))
        assert any(required_fragment in value for value in values), (
            f"{sample_path} should keep the solution format instruction"
        )
