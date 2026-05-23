# Kwai-Klear SWE-smith mini_swe_agent_plus Trajectories 66k

## Description

This dataset contains issue-solving trajectories collected with
[mini-swe-agent-plus](https://github.com/Kwai-Klear/mini-swe-agent-plus) on issues derived
from [SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith). Each trajectory
records an agent's iterative shell-based process for inspecting a repository, editing files,
running verification commands, and submitting a final diff.

## Dataset Information

**Source URL**: https://huggingface.co/datasets/Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k

**License**: MIT

**Split used**: `train`

**Size**: 65,994 trajectories

## Schema Mapping

- Raw `system` messages are skipped because they describe the mini-swe-agent-plus response format.
- Raw `user` messages containing task instructions are converted to user `TextObservation` entries, with formatting-only instructions removed while preserving the submission protocol.
- Raw `user` messages containing `<returncode>` and `<output>` tags are converted to environment `TextObservation` entries.
- Raw `assistant` messages are converted to bash `CodeAction` entries by extracting the last fenced `bash` block; preceding text is preserved as the action description.
- Trajectories are kept when the final assistant command includes `MINI_SWE_AGENT_FINAL_OUTPUT`, then a standard successful completion message and `finish` action are appended.
