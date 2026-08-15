# Jupyter Agent Dataset

## Description

[Jupyter Agent Dataset](https://huggingface.co/datasets/jupyter-agent/jupyter-agent-dataset) contains synthetic Jupyter notebook agent trajectories derived from real Kaggle notebooks and linked datasets. Each example includes a dataset-grounded question, a verified short answer, ChatML-style messages, Jupyter code execution tool calls, and execution observations.

The dataset is intended for training code/data-analysis agents that read notebook and dataset context, execute Python in a stateful Jupyter environment, and answer questions grounded in tabular or notebook data.

## Dataset Information

**Source URL**: https://huggingface.co/datasets/jupyter-agent/jupyter-agent-dataset

**Source code**: https://github.com/huggingface/jupyter-agent

**License**: Apache-2.0

**Size and split used**: The Hugging Face dataset card lists two splits, `thinking` and `non_thinking`, with 51,389 examples each. This ADP converter samples the `non_thinking` split by default because it contains the same tool-call structure without additional thinking tags.

**Sample size**: 3 trajectories in `sample_raw.json`, `sample_std.json`, and `sample_sft/openhands_v0.json`.

## Schema Mapping

- Raw `user` messages become `TextObservation` objects with `source="user"`.
- Raw `assistant` messages that call `add_and_execute_jupyter_code_cell` become Python `CodeAction` objects. The assistant prose is preserved as the action description.
- Raw `tool` messages become `TextObservation` objects with `source="environment"`.
- Raw `assistant` messages that call `final_answer` become `MessageAction` objects containing `<finish> ... </finish>` so the shared OpenHands SFT converter emits a `finish` tool call.
- Dataset metadata such as `question`, `answer`, `executor_type`, `files_used`, and `packages_used` is preserved in trajectory `details`.

## Reproduction

```bash
export MY_DATASET=jupyter-agent-dataset
export PYTHONPATH=`pwd`:$PYTHONPATH

MAX_ROWS=3 MAX_ORIGINAL_NOTEBOOK_CHARS=500000 python datasets/$MY_DATASET/extract_raw.py \
  | python scripts/jsonl_to_json.py \
  > datasets/$MY_DATASET/sample_raw.json

cat datasets/$MY_DATASET/sample_raw.json \
  | python scripts/json_to_jsonl.py \
  | python datasets/$MY_DATASET/raw_to_atif.py \
  | python scripts/jsonl_to_json.py \
  > datasets/$MY_DATASET/sample_atif.json
cat datasets/$MY_DATASET/sample_atif.json \
  | python scripts/json_to_jsonl.py \
  | python datasets/$MY_DATASET/atif_to_std.py \
  | python scripts/jsonl_to_json.py \
  > datasets/$MY_DATASET/sample_std.json

mkdir -p datasets/$MY_DATASET/sample_sft
cat datasets/$MY_DATASET/sample_std.json \
  | python scripts/json_to_jsonl.py \
  | python agents/openhands_v0/std_to_sft.py --is_web=no --api_env=execute_ipython_cell \
  | python scripts/jsonl_to_json.py \
  > datasets/$MY_DATASET/sample_sft/openhands_v0.json
```

## Known Limitations

- The converter currently targets the `non_thinking` split. Set `JUPYTER_AGENT_SPLIT=thinking` to extract the thinking split.
- The extraction script streams with the Hugging Face `datasets` package when available and falls back to the Hugging Face dataset viewer API for small samples.
- The original Kaggle files are referenced by metadata but are not included in this repository.

## Citation

```bibtex
@misc{jupyteragentdataset,
      title={Jupyter Agent Dataset},
      author={Baptiste Colle and Hanna Yukhymenko and Leandro von Werra},
      year={2025}
}
```
