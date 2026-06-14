# SALT-NLP/SWE-chat

## Description

[SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat) captures real-world AI coding-agent sessions from public GitHub repositories that use the Entire.io checkpointing CLI. The dataset includes full transcripts with user prompts, assistant responses, thinking traces, tool calls, tool results, metadata events, and linked git-history/code-attribution tables.

## Dataset Information

- **Source URL**: https://huggingface.co/datasets/SALT-NLP/SWE-chat
- **Project website**: https://swe-chat.com/
- **Paper**: https://arxiv.org/abs/2604.20779
- **Split used**: `train`
- **Configuration used for ADP conversion**: `conversations`
- **Size**: 2,692,480 conversation rows, 5,851 sessions, and 5,851 raw transcript files according to the Hugging Face dataset card
- **License**: ODC-BY (`odc-by` on the Hugging Face dataset card)
- **Access note**: The Hugging Face repository is public but gated. Users must accept the dataset conditions and provide a Hugging Face token with access when running `extract_raw.py`.

## Conversion Notes

- `extract_raw.py` streams the `conversations` table and emits one raw ADP item per `session_id`, preserving source conversation rows under `turns`.
- The extractor also attempts to load the smaller `sessions` table and stores the raw session row under `session` when available; `raw_to_atif.py` owns the later interpretation of session metadata.
- Rows with `turn_type="user_prompt"` are converted to `TextObservation(source="user")`.
- Rows with `turn_type="assistant_response"` are converted to `MessageAction`.
- Rows with `turn_type="assistant_thinking"` are converted to the OpenHands-compatible `think` API action.
- Shell-style tools (`Bash`, `run_command`, `shell`, `execute_bash`) are converted to `CodeAction(language="bash")`.
- File read/write/edit tools are mapped to the OpenHands `str_replace_editor` API when their paths and edit fields can be recovered.
- Other source-specific tools are preserved with a dataset-specific `generic_tool` API action containing the original tool name and parsed tool input.
- Tool-result rows and post-prompt metadata rows are converted to `TextObservation(source="environment")`; leading metadata before the first user prompt is skipped so SFT trajectories begin with a human task.
- `available_apis` is intentionally not populated because SWE-chat does not specify per-session tool availability; tool availability is inferred only from observed tool calls.

## Sample Generation

```bash
export MY_DATASET=SALT-NLP_SWE-chat
export PYTHONPATH=`pwd`:$PYTHONPATH
export HF_TOKEN=your_huggingface_token_with_swe_chat_access

python datasets/$MY_DATASET/extract_raw.py | head -3 | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_raw.json
cat datasets/$MY_DATASET/sample_raw.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/raw_to_atif.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_atif.json
cat datasets/$MY_DATASET/sample_atif.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/atif_to_std.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_std.json
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/openhands/std_to_sft.py --is_web=no --api_env=execute_bash | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft.json
```

For very large sessions during local experimentation, set `SWE_CHAT_MAX_TURNS_PER_SESSION` to cap the number of turns emitted per session. Do not use that cap for full reproducible extraction.
