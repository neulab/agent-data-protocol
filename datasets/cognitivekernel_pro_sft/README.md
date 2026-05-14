# CognitiveKernel-Pro-SFT

## Description

[CognitiveKernel/CognitiveKernel-Pro-SFT](https://huggingface.co/datasets/CognitiveKernel/CognitiveKernel-Pro-SFT) contains supervised fine-tuning data for the paper [Cognitive Kernel-Pro: A Framework for Deep Research Agents and Agent Foundation Models Training](https://huggingface.co/papers/2508.00414). The dataset consists of single-step action-module examples where a system prompt defines available CognitiveKernel tools, the user message provides the target task plus recent progress state, and the assistant emits a `Thought:` plus a fenced Python `Code:` action.

The released files cover URLQA (`ck-pro-web.sft.jsonl`), DocBench (`docbench.sft.jsonl`), TableBench (`tablebench.sft.jsonl`), and WebWalkerQA (`webwalker_subset.sft.jsonl`). Other sources mentioned by the authors are not included in the public release because of license restrictions.

## Dataset Information

- **Source URL**: https://huggingface.co/datasets/CognitiveKernel/CognitiveKernel-Pro-SFT
- **License**: `other` / `cognitive-kernel-pro`; the license grants use for academic purposes and prohibits commercial or production use. See the source dataset `LICENSE` for full terms.
- **Size / split used**: Hugging Face dataset server reports one `train` split with 47,271 examples. Source file sizes total about 874 MB of JSONL downloads.
- **Files used**: `ck-pro-web.sft.jsonl`, `docbench.sft.jsonl`, `tablebench.sft.jsonl`, and `webwalker_subset.sft.jsonl`.

## Schema Mapping

- Raw `messages` are modeled as role/content chat messages plus ADP-added `id`, `source_file`, and `source_index` fields.
- Raw `system` content is preserved by prepending it to the first `TextObservation(source="user")`, because it defines the CognitiveKernel action functions needed to understand the Python snippet.
- Raw `user` content becomes the task/progress portion of the same `TextObservation(source="user")`.
- Raw assistant messages matching the `Thought:` plus fenced Python `Code:` format become `CodeAction(language="python")`, with the thought stored as the action description.
- If an assistant message does not match the expected Thought/Code format, it falls back to `MessageAction` to preserve the raw content.

## Sample Generation

The sample files were generated from five representative records selected by `extract_raw.py --sample`, covering web search, web-agent use, file loading, and final `stop(...)` actions:

```bash
export MY_DATASET=cognitivekernel_pro_sft
export PYTHONPATH=`pwd`:$PYTHONPATH

python datasets/$MY_DATASET/extract_raw.py --sample | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_raw.json
cat datasets/$MY_DATASET/sample_raw.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/raw_to_standardized.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_std.json
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/openhands/std_to_sft.py --is_web=no --api_env=execute_ipython_cell | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft.json
```

## Known Limitations

- The source dataset is already SFT-style next-action data rather than full environment rollouts. Each ADP trajectory therefore contains the prompt state and the assistant's next Python code action, not an entire multi-step episode.
- CognitiveKernel tool functions such as `simple_web_search`, `web_agent`, `load_file`, and `stop` are preserved inside Python `CodeAction` snippets rather than converted to ADP `ApiAction` entries, because the raw assistant action is a Python code block that may combine computation with calls to those predefined functions.
