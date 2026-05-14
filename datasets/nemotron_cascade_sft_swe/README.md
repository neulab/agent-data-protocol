# Nemotron Cascade SFT SWE Dataset

## Description

Nemotron-Cascade-SFT-SWE is NVIDIA's supervised fine-tuning data for software engineering tasks in the Nemotron-Cascade training pipeline. The dataset combines examples derived from SWE-Bench-Train, SWE-reBench, SWE-Smith, R2E-Gym/R2E-Gym-Subset, and SWE-Fixer-Train-110K, excluding repositories present in SWE-Bench Verified to reduce evaluation contamination.

The prompts follow the agentless mini framework and cover three SWE task families:

- `SWE Localization`: identify files relevant to a GitHub issue.
- `SWE Repair`: produce a code patch for a GitHub issue using provided file snippets.
- `SWE TestGen`: generate a test case that reproduces or verifies a GitHub issue.

Each raw record contains a short two-turn conversation: a user prompt and a DeepSeek-R1-0528 assistant response, often including a `<think>...</think>` reasoning block.

## Dataset Information

**Source URL (Hugging Face)**: https://huggingface.co/datasets/nvidia/Nemotron-Cascade-SFT-SWE

**License**: CC-BY-4.0

**Split used**: `train` from the default configuration.

**Approximate size**: Hugging Face dataset viewer metadata reports 162,262 rows for the default train split, with an estimated row count of 141,244 at the time this converter was added. The dataset card also lists per-source sample counts for the three task families.

**Files used**:

- `swe_localization.jsonl`
- `swe_repair.jsonl`
- `swe_testgen.jsonl`

## Schema Mapping

- Raw `messages` with `role == "user"` become `TextObservation(source="user")`.
- Raw `messages` with `role == "assistant"` become `MessageAction` events.
- Assistant `<think>...</think>` content is preserved in the message text and also copied into `reasoning_content` when present.
- Dataset metadata (`category`, `source`, `generator`, and `thinking`) is stored in trajectory `details`.

Code blocks in assistant answers are treated as natural language response content rather than `CodeAction` events because the dataset is prompt/response SFT data, not an executed agent trajectory with shell observations.
