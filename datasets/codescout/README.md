# CodeScout Dataset

## Description

CodeScout contains code-localization rollouts from CodeScout models trained on SWE-Smith style tasks. The trajectories are OpenAI-chat-style conversations in which an assistant uses a terminal tool to inspect a repository and then returns the relevant files, classes, or functions through a localization finish action.

This integration combines representative samples from two Hugging Face dataset releases referenced in issue #184:

- `OpenHands/CodeScout_Training_Rollouts`: RL post-training rollouts from the CodeScout 4B and 14B models.
- `adityasoni17/CodeScout14B_RFT_SWE_Smith`: CodeScout 14B rollouts used to warm-start the 1.7B model with RFT.

The standardized conversion maps terminal tool calls to ADP `CodeAction` entries with `language="bash"`, maps terminal outputs to `TextObservation(source="environment")`, and maps `localization_finish` tool calls to a final `MessageAction` using the OpenHands `<finish> ... </finish>` convention.

## Dataset Information

- Source URLs:
  - https://huggingface.co/datasets/OpenHands/CodeScout_Training_Rollouts
  - https://huggingface.co/datasets/adityasoni17/CodeScout14B_RFT_SWE_Smith
- License: Not specified in the Hugging Face dataset cards.
- Size and splits used:
  - `OpenHands/CodeScout_Training_Rollouts`, config `CodeScout_4B`, train split: 15,805 examples.
  - `OpenHands/CodeScout_Training_Rollouts`, config `CodeScout_14B`, train split: 39,040 examples.
  - `adityasoni17/CodeScout14B_RFT_SWE_Smith`, default config, train split: 4,026 examples.
  - `adityasoni17/CodeScout14B_RFT_SWE_Smith`, default config, test split: 448 examples. The sample uses the train split.

## Conversion Notes

- System prompts are omitted from the standardized trajectory because the OpenHands SFT converter supplies the target system prompt.
- Assistant `terminal` tool calls are serialized as bash `CodeAction`s.
- Assistant `localization_finish` tool calls are serialized as finish `MessageAction`s containing the submitted locations JSON.
- Tool observations are paired with their matching tool call IDs so parallel terminal calls become ordered action/observation pairs in ADP.
- The raw rollouts include a tool-response echo after `localization_finish` that just repeats the submitted locations. Real OpenHands sessions terminate immediately on finish, so the echo is dropped to match that pattern (the locations are already captured in the finish action).
- Raw examples without an `instance_id` use deterministic source/split/row identifiers.
- The upstream `reward_dict.multilevel_localization_f1_reward` is attached to the trajectory's final content item as its `reward` value, since localization is single-step from a reward perspective. The other component rewards (`file_reward`, `module_reward`, `entity_reward`, `multiturn_reward`) and the per-instance tool schema are not propagated because they are not per-trajectory signals the standardized schema needs to carry.

## Sample Generation

```bash
export MY_DATASET=codescout
export PYTHONPATH=`pwd`:$PYTHONPATH

# One row per source/config keeps the committed sample small while covering both CodeScout releases.
MAX_ROWS_PER_SOURCE=1 python datasets/codescout/extract_raw.py \
  | python scripts/jsonl_to_json.py > datasets/codescout/sample_raw.json

cat datasets/codescout/sample_raw.json \
  | python scripts/json_to_jsonl.py \
  | python datasets/codescout/raw_to_atif.py \
  | python scripts/jsonl_to_json.py > datasets/codescout/sample_atif.json
cat datasets/codescout/sample_atif.json \
  | python scripts/json_to_jsonl.py \
  | python datasets/codescout/atif_to_std.py \
  | python scripts/jsonl_to_json.py > datasets/codescout/sample_std.json

mkdir -p datasets/codescout/sample_sft
cat datasets/codescout/sample_std.json \
  | python scripts/json_to_jsonl.py \
  | python agents/openhands_v0/std_to_sft.py --is_web=no --api_env=execute_bash \
  | python scripts/jsonl_to_json.py > datasets/codescout/sample_sft/openhands_v0.json
```
