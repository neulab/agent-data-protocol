# hybrid-gym

## Description

This dataset adds Agent Data Protocol support for the OpenHands-style trajectories published by the Hybrid Gym project on Hugging Face. The trajectories contain coding-agent conversations for four Hybrid Gym benchmark tasks: issue localization, function localization, function generation, and dependency search.

## Dataset Information

- **Source organization**: https://huggingface.co/hybrid-gym
- **Trajectory sources and split used**:
  - `hybrid-gym/issue_localize_1978i`, `train`, 1,978 trajectories
  - `hybrid-gym/func_localize_1438i`, `train`, 1,438 trajectories
  - `hybrid-gym/func_gen_552i`, `train`, 552 trajectories
  - `hybrid-gym/dep_search_502i`, `train`, 502 trajectories
- **Total size**: 4,470 trajectories across the four trajectory repositories according to the Hugging Face dataset cards
- **Format**: Parquet datasets with a `messages` column containing `{role, content}` chat turns
- **License**: The trajectory dataset cards do not declare a license. The companion raw Hybrid Gym benchmark datasets (`hybrid_gym_*_raw`) declare MIT on their Hugging Face cards.

## Conversion Notes

- `extract_raw.py` streams all four trajectory repositories in a deterministic round-robin order so small samples include multiple Hybrid Gym tasks.
- Raw `system` messages are preserved in `sample_raw.json` for fidelity but omitted from standardized trajectories because ADP does not model system prompts as content or details metadata.
- Raw `user` messages become `TextObservation(source="user")` unless they directly follow a parsed tool action, in which case they become `TextObservation(source="environment")` to represent tool output.
- Assistant messages with OpenHands `<function=...>` blocks are parsed into structured ADP actions:
  - `execute_bash` becomes `CodeAction(language="bash")`.
  - `think` and `str_replace_editor` become `ApiAction` instances.
  - `finish` becomes a terminal `MessageAction` with `<finish>...</finish>` content so the shared OpenHands v0 SFT converter emits the canonical finish function call.
- Assistant prose without function calls remains `MessageAction`; prose before a parsed function call is stored as the action description.
- The dataset includes an `api.py` containing OpenHands-compatible stubs for `think` and `str_replace_editor`, allowing validation and SFT conversion to use the shared OpenHands v0 converter without a dataset-local converter.
