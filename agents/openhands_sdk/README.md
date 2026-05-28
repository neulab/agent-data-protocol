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
