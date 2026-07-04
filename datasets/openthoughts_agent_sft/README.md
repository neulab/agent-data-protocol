# OpenThoughts-Agent SFT

OpenThoughts-Agent SFT contains terminal-agent conversations from the OpenThoughts-Agent project. The default extractor streams the newer and larger `open-thoughts/OpenThoughts-Agent-SFT-100K` dataset; the smaller `open-thoughts/OpenThoughts-Agent-v1-SFT` variant can be selected with `--variant v1-sft`.

- Source project: <https://github.com/open-thoughts/OpenThoughts-Agent>
- Default source dataset: <https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-100K>
- Optional smaller source dataset: <https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT>
- License: Apache-2.0

## Schema mapping

Raw rows contain OpenAI-style `conversations` plus task, run, model, and agent metadata.

- `system` messages become ATIF `system` steps.
- `user` messages become ATIF `user` steps.
- `assistant` messages become ATIF `agent` steps.
- Source metadata is preserved in `extra.raw`.

## Reproducing samples

```bash
export MY_DATASET=openthoughts_agent_sft
export PYTHONPATH=`pwd`:$PYTHONPATH

python datasets/$MY_DATASET/extract_raw.py --limit 4 | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_raw.json
cat datasets/$MY_DATASET/sample_raw.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/raw_to_atif.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_atif.json
cat datasets/$MY_DATASET/sample_atif.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/atif_to_std.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_std.json
mkdir -p datasets/$MY_DATASET/sample_sft
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/openhands_sdk/std_to_sft.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft/openhands_sdk.json
```
