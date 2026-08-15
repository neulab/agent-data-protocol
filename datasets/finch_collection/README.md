# Finch Collection

Finch Collection is a code-optimization and evolutionary-search dataset from MinnesotaNLP. Rows contain multi-turn trajectories for improving programs, along with parent and child code, metrics, and iteration metadata.

- Source dataset: <https://huggingface.co/datasets/minnesotanlp/Finch-Collection>
- License: CC-BY-4.0
- Split used: `train`
- Size: 217,780 rows

## Schema mapping

Raw rows contain a `trajectory` list in OpenAI-style chat format plus Finch-specific evolutionary-search metadata.

- `system` messages become ATIF `system` steps.
- `user` messages become ATIF `user` steps.
- `assistant` messages become ATIF `agent` steps.
- Assistant code blocks are preserved as message text, not converted to terminal tool calls, because Finch records program-evolution proposals rather than executed terminal sessions.
- `instance_uid` is used as the ADP row ID. `global_uid` is preserved in `extra.raw` but is not unique across rows.
- Finch metadata, metrics, code fields, and token lengths are preserved in `extra.raw`.

## Known limitations

- Rows are mostly single-turn system/user/assistant optimization examples with no structured tool calls or execution observations.
- Parent/child metrics and program text are preserved in `extra.raw`; they are not promoted to standardized top-level reward fields because the dataset stores them as task-specific JSON strings.

## Reproducing samples

```bash
export MY_DATASET=finch_collection
export PYTHONPATH=`pwd`:$PYTHONPATH

python datasets/$MY_DATASET/extract_raw.py --limit 4 | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_raw.json
cat datasets/$MY_DATASET/sample_raw.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/raw_to_atif.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_atif.json
cat datasets/$MY_DATASET/sample_atif.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/atif_to_std.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_std.json
mkdir -p datasets/$MY_DATASET/sample_sft
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/openhands_sdk/std_to_sft.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft/openhands_sdk.json
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/openhands_v0/std_to_sft.py --is_web=no --api_env=execute_bash | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft/openhands_v0.json
```
