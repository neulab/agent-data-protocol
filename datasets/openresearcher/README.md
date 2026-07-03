# OpenResearcher Dataset

## Description

OpenResearcher-Dataset is a large collection of deep-research agent trajectories released by OpenResearcher. Each trajectory contains a user research question, the agent's interleaved reasoning, calls to a browser search/open/find tool, tool observations, and a final answer.

The Hugging Face dataset is organized into seed configurations (`seed_42` through `seed_57`) with one `train` split per configuration. This converter preserves successful trajectories as browser-tool interaction traces.

## Dataset Information

**Source URL**: https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Dataset

**License**: MIT

**Size / split used**: The full dataset contains 16 `train` configurations with approximately 97.6K rows total. The committed samples are generated from the `seed_42` `train` split.

## Schema Mapping

- Raw `developer` and first `user` messages are combined into an initial `TextObservation` with `source="user"`.
- Assistant analysis text is stored as the `description` of the following browser `ApiAction` when possible.
- Assistant code messages addressed to `browser.search`, `browser.open`, and `browser.find` become `ApiAction` events using the dataset-local API signatures.
- Raw browser `tool` messages that include a rendered URL become `WebObservation` events; fetch errors and non-browser tool outputs remain `TextObservation(source="environment")` events.
- Assistant final-channel messages become `MessageAction` finish events.

## Sample Generation

```bash
export MY_DATASET=openresearcher
export PYTHONPATH=`pwd`:$PYTHONPATH

python datasets/openresearcher/extract_raw.py --configs seed_42 --offset 1 --limit 3 --max-messages 120 | python scripts/jsonl_to_json.py > datasets/openresearcher/sample_raw.json
cat datasets/openresearcher/sample_raw.json | python scripts/json_to_jsonl.py | python datasets/openresearcher/raw_to_atif.py | python scripts/jsonl_to_json.py > datasets/openresearcher/sample_atif.json
cat datasets/openresearcher/sample_atif.json | python scripts/json_to_jsonl.py | python datasets/openresearcher/atif_to_std.py | python scripts/jsonl_to_json.py > datasets/openresearcher/sample_std.json
mkdir -p datasets/openresearcher/sample_sft
cat datasets/openresearcher/sample_std.json | python scripts/json_to_jsonl.py | python agents/openhands_v0/std_to_sft.py --is_web=no --api_env=browser | python scripts/jsonl_to_json.py > datasets/openresearcher/sample_sft/openhands_v0.json
```
