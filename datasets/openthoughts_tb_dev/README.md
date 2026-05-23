# OpenThoughts-TB-Dev Dataset

## Description

OpenThoughts-TB-Dev is a development benchmark from the OpenThoughts-Agent project for terminal and shell-based agent tasks. The benchmark contains diverse tasks requiring agents to understand natural-language instructions, execute shell commands, and interact with files in a Linux environment.

This ADP conversion represents each task with a source instruction and the dataset-provided `solution/solve.sh` reference solution as a bash action. Tasks whose `solution/solve.sh` is an explicit placeholder (`no solution written`) are skipped because they do not contain a usable reference action for SFT conversion.

## Dataset Information

- Source URL: https://huggingface.co/datasets/open-thoughts/OpenThoughts-TB-dev
- License: Apache-2.0
- Size/split used: 70 tasks in the Hugging Face repository; 36 tasks include non-placeholder reference solution scripts and are emitted by `extract_raw.py`.
- Tags: agents, terminal, code, benchmark

## Schema Mapping

- `instruction.md` is converted to a user `TextObservation`.
- `solution/solve.sh` is converted to a bash `CodeAction`.
- Included verifier file names are summarized as an environment `TextObservation`.
- A final `MessageAction` marks task completion.

## Known Limitations

This source is a benchmark task repository rather than a multi-turn trajectory dataset. The converter therefore captures an oracle single-action trajectory from the reference solution script instead of interactive agent execution logs.

The raw source files include Terminal-Bench contamination canary comments. These lines are preserved in `sample_raw.json` to reflect the source data, but `raw_to_standardized.py` strips them from generated `CodeAction` content so standardized and SFT training samples do not carry benchmark canary markers.
