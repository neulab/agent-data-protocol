# EnterpriseLab

## Description

EnterpriseLab is an enterprise-agent benchmark containing ReAct-style tool-use
conversations over realistic workplace applications such as GitLab,
Rocket.Chat, OwnCloud, ERP/accounting systems, and project-management APIs.
The local source file used for this ADP integration is
`enterprise_arena_gold.json` from the EnterpriseLab repository.

The dataset focuses on:
- Enterprise workflow automation tasks
- Multi-application tool use
- OpenAI-style chat messages with assistant tool calls and tool observations
- Successful and unsuccessful enterprise task trajectories

## Dataset Information

- Source repository: https://github.com/ast-fri/EnterpriseLab
- Source file used locally: `enterprise_arena_gold.json`
- Related ADP paper: https://arxiv.org/abs/2510.24702
- License: See the upstream EnterpriseLab repository
- Split used: gold trajectories from `enterprise_arena_gold.json`

## Schema Mapping

- Raw `system`, `user`, and `assistant` messages are preserved as ATIF steps.
- Raw assistant `tool_calls` become ATIF `ToolCall` entries with the original
  function names and JSON arguments.
- Raw `tool` messages become ATIF observations and are linked to the preceding
  assistant tool call when possible.
- `atif_to_std.py` applies ADP's shared tool-call normalization while retaining
  EnterpriseLab-specific tools declared in `metadata.json`.
- `sample_sft/openhands_sdk.json` is generated from the standardized ATIF sample
  through the shared OpenHands SDK converter.
- `sample_sft/sweagent.json` is generated from the same standardized sample through
  the shared SWE-Agent converter. Custom EnterpriseLab tools are wrapped as SWE-Agent
  `bash` commands (for example, `search_repositories(search=keystone)`).
- `sample_sft/agentlab.json` is **not** generated for EnterpriseLab. AgentLab targets
  web-browsing trajectories with accessibility trees and browser actions; EnterpriseLab
  uses enterprise API tool calls instead.

## Regenerating Samples

From the repository root:

```bash
export MY_DATASET=enterpriselab
export PYTHONPATH=`pwd`:$PYTHONPATH

# Extract raw data (5 samples)
python datasets/$MY_DATASET/extract_raw.py --source enterprise_arena_gold.json | head -5 | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_raw.json

# Convert raw to ATIF
cat datasets/$MY_DATASET/sample_raw.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/raw_to_atif.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_atif.json

# Convert ATIF to standardized
cat datasets/$MY_DATASET/sample_atif.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/atif_to_std.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_std.json

# Convert to OpenHands SDK SFT
mkdir -p datasets/$MY_DATASET/sample_sft
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/openhands_sdk/std_to_sft.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft/openhands_sdk.json

# Convert to SWE-Agent SFT
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/sweagent/std_to_sft.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft/sweagent.json
```

`metadata.json` includes only tools that appear in `tool_calls` across the source
file, with parameter names inferred from those calls. OpenHands SDK requires
explicit parameter names in `metadata.json`.

AgentLab SFT (`sample_sft/agentlab.json`) is **not applicable** to EnterpriseLab.
The AgentLab converter expects web-browsing trajectories with browser actions and
accessibility trees; EnterpriseLab uses enterprise API tool calls instead.

Validate the dataset:

```bash
python -m pytest tests/test_dataset_structure.py -v -k enterpriselab
python -m pytest tests/test_raw_schemas.py -v -k enterpriselab
python -m pytest tests/test_standardized_schemas.py -v -k enterpriselab
```

## Citation

Please cite EnterpriseLab using the citation requested by the upstream
EnterpriseLab authors.
