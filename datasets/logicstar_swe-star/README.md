# LogicStar/SWE-Star

## Description

[SWE-Star](https://huggingface.co/datasets/LogicStar/SWE-Star) contains approximately 244k OpenHands-style software engineering trajectories distilled from Devstral-2-Small on SWE-Smith tasks. The trajectories use XML tool calls from a custom OpenHands-like scaffold with `think`, `str_replace_editor`, `execute_bash`, and terminal submission tools.

## Dataset Information

- **Source URL**: https://huggingface.co/datasets/LogicStar/SWE-Star
- **Project repository**: https://github.com/logic-star-ai/swe-star
- **Split used**: `train`
- **Size**: 244,025 rows in Hugging Face dataset-server metadata
- **License**: Not specified on the Hugging Face dataset card or project repository at the time this converter was added
- **Format**: Parquet rows with `timestamp`, `instance_id`, `exit_status`, `stitched`, `full`, `result`, and `resolved` fields. The `stitched` and `full` fields are JSON-encoded message lists.

## Conversion Notes

- `stitched` is used for ADP conversion because it preserves the trajectory turns while keeping observations shorter than the `full` trace.
- System prompts are skipped from the ADP content and retained implicitly by the OpenHands SFT system prompt.
- The first user turn is converted to `TextObservation(source="user")`.
- Subsequent `EXECUTION RESULT` user turns are converted to `TextObservation(source="environment")` with the execution-result prefix removed, allowing the OpenHands SFT converter to reconstruct it from the preceding action.
- Assistant XML calls to `execute_bash` are converted to `CodeAction(language="bash")`.
- Assistant XML calls to `str_replace_editor` and `think` are converted to `ApiAction`.
- Assistant XML calls to `finish` or `submit` are converted to terminal `MessageAction` content using the ADP `<finish> ... </finish>` convention.
- Only resolved trajectories are emitted by `extract_raw.py` and `raw_to_standardized.py` for supervised fine-tuning samples.

## Sample Generation

```bash
export MY_DATASET=logicstar_swe-star
export PYTHONPATH=`pwd`:$PYTHONPATH

python datasets/$MY_DATASET/extract_raw.py | head -3 | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_raw.json
cat datasets/$MY_DATASET/sample_raw.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/raw_to_standardized.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_std.json
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/openhands/std_to_sft.py --is_web=no --api_env=execute_bash | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft.json
```
