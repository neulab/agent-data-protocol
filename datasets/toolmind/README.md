# ToolMind

ToolMind is a large-scale, reasoning-enhanced tool-use dataset from Nanbeige. The Hugging Face dataset card describes 160k synthetic data instances generated with more than 20k tools plus 200k augmented open-source instances, for roughly 360k total examples. The dataset is distributed under the Apache-2.0 license.

- Source: <https://huggingface.co/datasets/Nanbeige/ToolMind>
- Paper: <https://arxiv.org/abs/2511.15718>
- License: Apache-2.0
- Config used: `test`
- Sample split/file used here: `graph_syn_datasets/graphsyn.jsonl`

## Schema mapping

Raw ToolMind rows contain a `conversations` array in OpenAI-style chat format plus a per-row `tools` array.

- `user` messages become `TextObservation(source="user")`.
- `system` messages are preserved as user-visible `TextObservation` values prefixed with `System instructions:` because the standardized schema does not define a `system` observation source.
- Assistant messages without tool calls become `MessageAction`; raw `<think>...</think>` blocks are moved to `reasoning_content` without the tags.
- Assistant `tool_calls` become `ApiAction`; non-thinking assistant text is stored as the action description, while raw `<think>...</think>` blocks are moved to `reasoning_content` without the tags.
- `tool` messages become `TextObservation(source="environment")`.
- Raw tool definitions are advertised on the top-level `Trajectory.available_apis` field as a list of identifiers. The dataset's `api.py` carries the matching stubs, and the OpenHands SFT converter uses `include_apis=trajectory.available_apis` to expose only the row-specific API signatures.

## Reproducing samples

```bash
export MY_DATASET=toolmind
export PYTHONPATH=`pwd`:$PYTHONPATH

python datasets/$MY_DATASET/extract_raw.py | head -4 | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_raw.json
cat datasets/$MY_DATASET/sample_raw.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/raw_to_atif.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_atif.json
cat datasets/$MY_DATASET/sample_atif.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/atif_to_std.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_std.json
mkdir -p datasets/$MY_DATASET/sample_sft
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/openhands_v0/std_to_sft.py --is_web=no --api_env=execute_bash | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft/openhands_v0.json
```
