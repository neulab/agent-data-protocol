# AllenAI Sera-4.6-Lite-T2 Dataset

## Description

Sera-4.6-Lite-T2 contains software engineering agent trajectories generated from the second rollout of SVG on 121 SWE-smith codebases using GLM-4.6 as the teacher model. The dataset includes full OpenAI-style message trajectories with tool calls, tool observations, generated rollout patches, target patches, and task metadata.

The trajectories capture agents working in `/testbed` repositories to resolve synthetic software engineering issues. They include repository inspection, file edits through `str_replace_editor`, shell commands through `bash`, and final `submit` actions.

## Dataset Information

**Source URL**: https://huggingface.co/datasets/allenai/Sera-4.6-Lite-T2

**License**: Open Data Commons Attribution License v1.0 (ODC-By)

**Size/Split Used**: 36,083 trajectories from the single JSONL file `sera-4.6-lite-t2_36083_string_enriched.jsonl`. The dataset card does not define multiple splits; the converter streams the JSONL file in source order.

## Schema Mapping

- Raw `system` messages are skipped.
- Raw `user` messages are converted to `TextObservation` with source `user`.
- Raw `tool` messages are converted to `TextObservation` with source `environment`; leading `OBSERVATION:` prefixes are removed.
- Assistant `bash` tool calls are converted to `CodeAction(language="bash")`.
- Assistant `str_replace_editor` and `submit` tool calls are converted to `ApiAction` entries backed by `api.py`.
- Assistant messages without tool calls are converted to `MessageAction`.
- If a trajectory does not already end with a finish message, the converter appends a deterministic completion observation and `<finish>` message.

## Known Limitations

The source README notes that T2 trajectories can be verified by comparing `rollout_patch` and `target_patch`, but verification was not performed in the dataset creators' main experiments. This converter preserves both patches in `details` and does not filter by patch equivalence.
