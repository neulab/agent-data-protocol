# GAIR daVinci-Dev Dataset

## Description

GAIR/daVinci-Dev is a mixed software-engineering dataset released with the paper "daVinci-Dev: Agent-native Mid-training for Software Engineering". The Hugging Face dataset contains contextually-native PR-derived records and environmentally-native executable rollouts. This ADP converter targets the `env_native` config (`env-native.jsonl`), which contains SWE-Agent + GLM-4.6 trajectories collected on SWE-rebench tasks.

The environment-native rows use chat-style messages with JSON tool calls. Assistant tool calls are mapped to ADP actions, and tool responses are mapped to environment observations.

## Dataset Information

**Source URL**: https://huggingface.co/datasets/GAIR/daVinci-Dev

**Upstream utilities**: https://github.com/GAIR-NLP/daVinci-Dev/tree/main/env_traj_utils

**License**: Mixed release. The environment-native subset is derived from SWE-rebench under CC-BY-4.0; the daVinci-Dev pipeline is Apache-2.0; the contextually-native subset includes repositories with permissive licenses. Downstream users should verify the license of underlying repositories and generated code.

**Size / split used**: Hugging Face reports the dataset as 1M-10M rows overall. This converter streams the `env_native` config, `train` split, from `env-native.jsonl`.

**Access note**: The Hugging Face repository is gated. `extract_raw.py` supports Hugging Face authentication through the standard `HF_TOKEN` environment variable or local Hugging Face login.

## Schema Mapping

| Raw field | ADP mapping |
| --- | --- |
| `messages[].role == "system"` | Skipped; first system prompt is retained in trajectory details. |
| `messages[].role == "user"` | `TextObservation(source="user")`. |
| `messages[].role == "assistant"` without `tool_calls` | `MessageAction`, preserving `reasoning_content` when present. |
| Assistant `tool_calls` named `bash`, `execute_bash`, or `terminal` | `CodeAction(language="bash")` using the `command` argument. |
| Assistant `tool_calls` named `submit` or `finish` | `MessageAction` containing an ADP finish marker. |
| Assistant `tool_calls` named `str_replace_editor` | `ApiAction(function="str_replace_editor")`. |
| `messages[].role == "tool"` | `TextObservation(source="environment")`. |

The committed sample focuses on the documented environment-native message format from the upstream conversion utilities: system/user/assistant/tool messages, JSON tool calls, bash execution, editor actions, tool observations, and submit-style termination.
