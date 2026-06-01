# NVIDIA SWE-Zero OpenHands Trajectories Dataset

## Description

SWE-Zero OpenHands Trajectories is an agentic instruction-tuning dataset for software engineering agents. It contains OpenHands trajectories synthesized with Qwen/Qwen3-Coder-480B-A35B-Instruct for SWE-Bench-style tasks sourced from permissively licensed software-engineering datasets.

## Dataset Information

- **Source URL**: https://huggingface.co/datasets/nvidia/SWE-Zero-openhands-trajectories
- **Technical report**: https://arxiv.org/abs/2604.01496
- **License**: CC BY 4.0, with source repositories limited by the dataset authors to MIT, Apache-2.0, BSD-2-Clause, and BSD-3-Clause licenses.
- **Split used**: `train`
- **Approximate size**: 318,115 trajectories over 118,092 issues according to the dataset card.
- **Source task datasets**: SWE-Gym, SWE-Gym-Raw, R2E-Gym-Subset, SWE-Fixer-Train-110K, and SWE-rebench.
- **Scaffolding**: OpenHands
- **Bootstrapping model**: Qwen/Qwen3-Coder-480B-A35B-Instruct

## Schema Mapping

The raw dataset is in chat-completion format:

- `system` messages are skipped during standardization.
- `user` messages become `TextObservation(source="user")`.
- `tool` messages become `TextObservation(source="environment")`.
- Assistant messages without tool calls become `MessageAction`.
- Assistant `execute_bash` tool calls become `CodeAction(language="bash")`.
- Assistant `finish` tool calls become `MessageAction` containing `<finish> ... </finish>` so OpenHands SFT conversion produces a terminal finish call.
- Other assistant tool calls, currently `think` and `str_replace_editor` in the sampled trajectories, become `ApiAction` entries with JSON arguments preserved.

The standardized trajectory details preserve the raw `instance_id`, repository, per-repository source license, original source dataset name, and final `model_patch`.
