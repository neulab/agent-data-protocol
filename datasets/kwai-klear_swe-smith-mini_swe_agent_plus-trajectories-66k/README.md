# Kwai-Klear SWE-smith mini-swe-agent-plus Trajectories 66k

## Description

This dataset contains approximately 66k issue-solving trajectories collected with `mini-swe-agent-plus` on software engineering issues derived from SWE-smith. Each trajectory records a system prompt, an initial issue/task prompt, assistant responses containing a thought plus one bash command, and shell execution observations.

## Dataset Information

**Source URL**: https://huggingface.co/datasets/Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k

**License**: MIT

**Split used**: `train`

**Size**: 65,994 trajectories

## Schema Mapping

- Raw `system` messages are omitted from standardized trajectories because the OpenHands SFT converter supplies the target system prompt.
- The first raw `user` message becomes a `TextObservation` with source `user`.
- Later raw `user` messages are command execution observations and become `TextObservation` entries with source `environment`.
- Raw `assistant` messages with a fenced bash command become `CodeAction(language="bash")`, with the preceding `THOUGHT:` text preserved as the action description.
- Raw assistant messages without a parseable command are preserved as `MessageAction` entries.
