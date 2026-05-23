# DTA-Tool Dataset

## Description

DTA-Tool is a tool-learning dataset introduced with the paper "Divide-Then-Aggregate: An Efficient Tool Learning Method via Parallel Tool Invocation". It is designed for training large language models to plan and execute one or more tool calls in parallel, then aggregate the tool results into a final answer.

The dataset is transformed from ToolBench and each trajectory contains a user query, an AutoGPT-style system prompt listing available APIs, assistant thoughts with `Function Call` JSON payloads, tool observations, and a final `Finish` call.

## Dataset Information

- **Source URL**: https://huggingface.co/datasets/dongsheng/DTA-Tool
- **Source file**: `DTA_tool.json`
- **License**: Not specified in the Hugging Face dataset card
- **Size / split used**: Hugging Face reports the dataset size category as `10K<n<100K`; the converter reads the single JSON file as one train-style stream.
- **Paper**: https://arxiv.org/abs/2501.12432

## Schema Mapping

- Raw `system` messages advertise the API functions available for that trajectory. Their identifiers are recorded on the top-level `Trajectory.available_apis` field, so the OpenHands SFT converter can filter `api.py` per-trajectory via `include_apis`. The full set of advertised signatures is mirrored as stubs in this dataset's `api.py`.
- Raw `user` messages become `TextObservation` entries with `source="user"`.
- Raw `assistant` messages with `Thought:` and `Function Call:` become one or more `ApiAction` entries, preserving parallel tool calls as consecutive actions with the same thought.
- Raw `assistant` `Finish` calls become `MessageAction` entries containing `<finish> ... </finish>` so the shared OpenHands SFT converter emits the standard `finish` tool call.
- Raw `function` messages that parse as dictionary payloads become `JsonObservation` entries with `source="environment"`; unparseable payloads fall back to `TextObservation`.

## Citation

```bibtex
@misc{zhu2025dividethenaggregateefficienttoollearning,
      title={Divide-Then-Aggregate: An Efficient Tool Learning Method via Parallel Tool Invocation},
      author={Dongsheng Zhu and Weixian Shi and Zhengliang Shi and Zhaochun Ren and Shuaiqiang Wang and Lingyong Yan and Dawei Yin},
      year={2025},
      eprint={2501.12432},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.12432},
}
```
