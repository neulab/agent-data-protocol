# OpenHands SDK SFT Converter

This converter builds SFT records through OpenHands SDK primitives instead of
manually formatting OpenAI chat messages.

For each standardized trajectory it:

1. Loads dataset `metadata.json`.
2. Registers every metadata `custom_tools` entry as an SDK custom tool.
3. Creates an SDK `Conversation`, letting the SDK construct the system prompt
   and tool map.
4. Appends converted standardized actions and observations to the SDK event log.
5. Uses SDK event/message formatting helpers to emit `messages` and `tools`.

Regenerate one dataset with:

```bash
export MY_DATASET=agenttuning_alfworld
export PYTHONPATH=`pwd`:$PYTHONPATH
python scripts/json_to_jsonl.py < datasets/$MY_DATASET/sample_std.json \
  | python agents/openhands_sdk/std_to_sft.py \
  | python scripts/jsonl_to_json.py \
  > datasets/$MY_DATASET/sample_sft/openhands_sdk.json
```

## Condensation prompt SFT utility

`condensation_sft.py` replays standardized ADP trajectories through the same
OpenHands SDK event builder and `LLMSummarizingCondenser` trigger logic. If a
trajectory produces `n` condensation prompts, the utility emits `n + 1` normal
OpenHands SDK trajectory records interleaved with those prompt-only SFT records,
matching the condensed-history states the SDK would expose between condensation
boundaries.

```bash
export MY_DATASET=agenttuning_os
export PYTHONPATH=`pwd`:$PYTHONPATH
python scripts/json_to_jsonl.py < datasets/$MY_DATASET/sample_std.json \
  | python agents/openhands_sdk/condensation_sft.py --max-tokens 2000 \
  > /tmp/openhands_sdk_with_condensation_prompts.jsonl
```

Use `--include-trajectories no` to emit only condensation prompt records. By
default, prompt records include only the condensation input. Add
`--condensation-output llm` to call `LLM_MODEL`/`LLM_API_KEY`/`LLM_BASE_URL` via
LiteLLM and append the generated summary as the assistant output; use
`--condensation-output placeholder` for deterministic test output without an
external LLM call. Token-triggered condensation uses the selected condenser
model's token counter, so set `--model`/`LLM_MODEL` to the model whose context
budget should determine condensation boundaries.
