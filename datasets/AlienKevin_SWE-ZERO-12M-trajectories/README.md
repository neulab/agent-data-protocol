# SWE-ZERO 12M Trajectories Dataset

## Description

SWE-ZERO 12M Trajectories is a large-scale execution-free agentic coding trace dataset. It contains mini-swe-agent v1 style shell trajectories sampled from real GitHub PR snapshots, intended for mid-training coding agents on repository navigation, editing, and bash-based tool use.

## Dataset Information

- **Source URL**: https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories
- **License**: Apache-2.0
- **Split used**: `train`
- **Approximate size**: 12,290,800 rollouts, 122,908 unique PRs, 3,222 repositories, 16 programming languages, and 112B tokens according to the dataset card.
- **Source task dataset**: nebius/SWE-rebench-V2-PRs
- **Trajectory format**: mini-swe-agent v1
- **Bootstrapping model**: ricdomolm/mini-coder-1.7b

## Schema Mapping

The raw dataset is a list of chat-style messages with `role` and `content` fields:

- `system` messages are skipped because they only define the mini-swe-agent response format and execution-free shell constraints.
- Initial `user` task messages become `TextObservation(source="user")`.
- Later `user` messages beginning with `Observation:` become `TextObservation(source="environment")` with the prefix removed.
- `assistant` messages containing fenced `bash` blocks become `CodeAction(language="bash")`; the text before the final bash block is preserved as the action description after removing a leading `THOUGHT:` label.
- `assistant` messages without a bash block become `MessageAction` entries so malformed or terminal natural-language turns are preserved.

The standardized trajectory details preserve the raw `instance_id`, repository, `trajectory_format`, `exit_status`, and `duration_sec`. Trajectory IDs are derived deterministically from the instance ID plus a content hash because the source dataset contains many independent rollouts per PR with the same `instance_id`.

## Known Limitations

The dataset card describes this corpus as a mid-training dataset rather than a verified SFT dataset. The trajectories are execution-free, not validated against tests, and many rollouts terminate with `incomplete` or other non-submitted statuses. This converter preserves those trajectories instead of filtering to submitted-only samples.
