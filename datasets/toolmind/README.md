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
- Assistant messages without tool calls become `MessageAction`.
- Assistant `tool_calls` become `ApiAction`; the assistant text is stored as the action description so reasoning traces such as `<think>...</think>` are preserved.
- `tool` messages become `TextObservation(source="environment")`.
- Raw tool definitions are converted into `details["available_apis"]` so the OpenHands SFT converter can expose the row-specific API signatures.

## Reproducing samples

```bash
export MY_DATASET=toolmind
export PYTHONPATH=`pwd`:$PYTHONPATH

python datasets/$MY_DATASET/extract_raw.py | head -4 | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_raw.json
cat datasets/$MY_DATASET/sample_raw.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/raw_to_standardized.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_std.json
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/openhands/std_to_sft.py --is_web=no --api_env=execute_bash | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft.json
```
