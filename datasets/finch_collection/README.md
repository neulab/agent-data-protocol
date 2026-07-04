# Finch Collection

Finch Collection is a code-optimization and evolutionary-search dataset from MinnesotaNLP. Rows contain multi-turn trajectories for improving programs, along with parent and child code, metrics, and iteration metadata.

- Source dataset: <https://huggingface.co/datasets/minnesotanlp/Finch-Collection>
- License: CC-BY-4.0

## Schema mapping

Raw rows contain a `trajectory` list in OpenAI-style chat format plus Finch-specific evolutionary-search metadata.

- `system` messages become ATIF `system` steps.
- `user` messages become ATIF `user` steps.
- `assistant` messages become ATIF `agent` steps.
- Finch metadata, metrics, code fields, and token lengths are preserved in `extra.raw`.

## Reproducing samples

```bash
export MY_DATASET=finch_collection
export PYTHONPATH=`pwd`:$PYTHONPATH

python datasets/$MY_DATASET/extract_raw.py --limit 4 | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_raw.json
cat datasets/$MY_DATASET/sample_raw.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/raw_to_atif.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_atif.json
cat datasets/$MY_DATASET/sample_atif.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/atif_to_std.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_std.json
mkdir -p datasets/$MY_DATASET/sample_sft
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/openhands_sdk/std_to_sft.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft/openhands_sdk.json
```
