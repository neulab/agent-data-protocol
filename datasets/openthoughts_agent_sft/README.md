# OpenThoughts-Agent SFT

OpenThoughts-Agent SFT contains terminal-agent conversations from the OpenThoughts-Agent project. The default extractor streams the newer and larger `open-thoughts/OpenThoughts-Agent-SFT-100K` dataset; the smaller `open-thoughts/OpenThoughts-Agent-v1-SFT` variant can be selected with `--variant v1-sft`.

- Source project: <https://github.com/open-thoughts/OpenThoughts-Agent>
- Default source dataset: <https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-100K>
- Optional smaller source dataset: <https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT>
- License: Apache-2.0
- Split used: `train`
- Size: 94,334 rows for `OpenThoughts-Agent-SFT-100K`; 15,209 rows for `OpenThoughts-Agent-v1-SFT`

## Schema mapping

Raw rows contain OpenAI-style `conversations` plus task, run, model, and agent metadata.

- The first raw `user` message contains both system instructions and the task. The converter splits it at `Task Description:` so the instructions become an ATIF `system` step and the task remains an ATIF `user` step.
- `user` messages become ATIF `user` steps.
- `assistant` messages become ATIF `agent` steps.
- Assistant JSON-command responses are parsed into terminal tool calls. Multiple `commands[*].keystrokes` entries in one JSON response remain multiple tool calls on the same ATIF step because the following raw terminal output is batched into one user turn.
- Source metadata is preserved in `extra.raw`.

## Known limitations

- The source batches multiple terminal commands and their combined outputs into single assistant/user turns. The converter preserves that batch structure instead of inventing intermediate observations that are not present in the raw data.
- Terminal output arrives as raw `user` messages rather than structured tool-result records, so the conversion cannot reliably link each output line to a specific command when a batch contains more than one command.

## Reproducing samples

```bash
export MY_DATASET=openthoughts_agent_sft
export PYTHONPATH=`pwd`:$PYTHONPATH

python datasets/$MY_DATASET/extract_raw.py --limit 4 | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_raw.json
cat datasets/$MY_DATASET/sample_raw.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/raw_to_atif.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_atif.json
cat datasets/$MY_DATASET/sample_atif.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/atif_to_std.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_std.json
mkdir -p datasets/$MY_DATASET/sample_sft
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/openhands_sdk/std_to_sft.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft/openhands_sdk.json
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/openhands_v0/std_to_sft.py --is_web=no --api_env=execute_bash | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft/openhands_v0.json
```
