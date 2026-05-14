# Nebius SWE-rebench OpenHands Trajectories Dataset

## Description

SWE-rebench-OpenHands-Trajectories is a dataset of multi-turn software-engineering agent trajectories collected with OpenHands v0.54.0 and Qwen/Qwen3-Coder-480B-A35B-Instruct. The tasks come from real GitHub issues in the Nebius SWE-rebench benchmark, and the trajectories capture assistant reasoning, tool calls, and environment observations while attempting issue resolution.

## Dataset Information

- **Source URL**: https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories
- **Source benchmark**: https://huggingface.co/datasets/nebius/SWE-rebench
- **License**: CC BY 4.0
- **Split used**: `train`
- **Approximate size**: 67,074 trajectories, including 32,161 successful trajectories according to the dataset card.
- **Scaffolding**: OpenHands v0.54.0
- **Bootstrapping model**: Qwen/Qwen3-Coder-480B-A35B-Instruct

## Schema Mapping

The raw dataset is in chat-completion format:

- `system` messages are skipped during standardization.
- `user` messages become `TextObservation(source="user")`.
- `tool` messages become `TextObservation(source="environment")`.
- Assistant messages without tool calls become `MessageAction`.
- Assistant `execute_bash` tool calls become `CodeAction(language="bash")`.
- Assistant `finish` tool calls become `MessageAction` containing `<finish> ... </finish>`.
- Other assistant tool calls, such as `think`, `str_replace_editor`, and `task_tracker`, become `ApiAction` entries with their JSON arguments preserved.

Only resolved trajectories are emitted by `extract_raw.py` and `raw_to_standardized.py`, matching the dataset's successful-trajectory subset for SFT conversion.
