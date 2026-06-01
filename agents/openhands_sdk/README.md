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

## Condensation SFT utility

`condensation_sft.py` replays standardized ADP trajectories through the same
OpenHands SDK event builder and `LLMSummarizingCondenser` trigger logic. If a
trajectory produces `n` condensations, the utility emits `n + 1` normal
OpenHands SDK trajectory records interleaved with `n` condenser SFT records that
include both the condensation prompt and the LLM-generated summary. This matches
the condensed-history states the SDK would expose between condensation
boundaries.

```bash
export MY_DATASET=agenttuning_os
export PYTHONPATH=`pwd`:$PYTHONPATH
export LLM_MODEL=gpt-4o-mini
export LLM_API_KEY=...
python scripts/json_to_jsonl.py < datasets/$MY_DATASET/sample_std.json \
  | python agents/openhands_sdk/condensation_sft.py --max-tokens 2000 \
  > /tmp/openhands_sdk_with_condensation_summaries.jsonl
```

The utility always calls the condenser LLM and does not support prompt-only or
placeholder-output generation. Use `--include-trajectories no` to emit only the
condenser SFT records. Token-triggered condensation uses the selected condenser
model's token counter, so set `--model`/`LLM_MODEL` to the model whose context
budget should determine condensation boundaries.
