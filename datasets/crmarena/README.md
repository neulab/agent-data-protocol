# CRMArena

## Description

CRMArena is a benchmark from Salesforce AI Research for evaluating LLM agents on
professional Customer Relationship Management (CRM) tasks in a realistic
Salesforce environment. Agents answer business questions (e.g. case routing,
knowledge QA, purchase history, agent performance analytics) by making
tool/function calls against a live Salesforce organization and reasoning over
the results.

The public `Salesforce/CRMArena` dataset ships **tasks and ground-truth
answers**, not agent trajectories. This ADP integration therefore uses
**agent rollouts**: the CRMArena tool-calling agent (`--agent_strategy tool_call`,
`--org_type original`) is run over the tasks against the Salesforce sandbox, and
each rollout's full message history (system prompt → user query → assistant tool
calls → tool observations → final `respond`) is converted into an ATIF
trajectory.

The dataset focuses on:
- CRM workflow automation and analytics tasks across all 9 original CRMArena task
  categories (`handle_time`, `transfer_count`, `knowledge_qa`,
  `best_region_identification`, `case_routing`, `named_entity_disambiguation`,
  `policy_violation_identification`, `top_issue_identification`,
  `monthly_trend_analysis`)
- Multi-turn Salesforce tool use (27 domain functions, incl. SOQL/SOSL queries)
- OpenAI-style chat messages with assistant tool calls and tool observations
- Both successful and unsuccessful task trajectories (including real tool errors
  and agent recovery/retries)

The committed sample contains one representative trajectory per task category
(9 trajectories, 6–11 steps and 4–9 tool calls each).

## Dataset Information

- Original paper: [CRMArena: Understanding the Capacity of LLM Agents to Perform Professional CRM Tasks in Realistic Environments](https://arxiv.org/abs/2411.02305)
- Original dataset: https://huggingface.co/datasets/Salesforce/CRMArena
- Original repository: https://github.com/SalesforceAIResearch/CRMArena
- Related ADP paper: https://arxiv.org/abs/2510.24702
- License: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0) — see [LICENSE](LICENSE)
- Split used: rollouts over the `test` split tasks (all 9 original task categories)
- Generation model: the rollouts committed here were produced with GPT-5.1 via an
  OpenAI-compatible endpoint (any tool-calling model works; trajectories are not
  deterministic across regenerations).
- Raw source used locally: CRMArena result logs
  (`results/original_toolcall/results_*_tool_call_<category>.json`) produced by the
  CRMArena tool-calling agent; each record's `traj` is an OpenAI-style message list.
- Regeneration deps for producing rollouts are listed in
  [`requirements.txt`](requirements.txt); the ADP conversion scripts themselves need
  only the standard library and shared repo dependencies.

## Tools

The 27 CRMArena functions are declared as `custom_tools` in
[`metadata.json`](metadata.json) with names, descriptions, and parameter schemas
taken from `crm_sandbox/env/functions.py` (`<function>.__info__`).
The agent's final answer is always emitted through the `respond` tool.

Two deviations from the raw `__info__` schemas, made so the tool calls that
actually appear in the trajectories validate against `metadata.json`:
- `search_knowledge_articles`: the upstream schema's `required` lists `keywords`,
  a name that is not among its `properties` (the sole property is `search_term`).
  The property `keywords` (an optional string array) is added and `required` is
  left empty, because agents in practice call the tool with `search_term`,
  `keywords`, or both (the underlying function only accepts `search_term`, so
  `keywords` calls surface as recoverable tool errors in the trajectories).

## Schema Mapping

- Each rollout record's `traj` (renamed to `messages` by `extract_raw.py`) is an
  OpenAI-style chat list and is converted by the shared
  `scripts/raw_to_atif_common.py` converter.
- Raw `system` / `user` messages become ATIF `system` / `user` steps.
- Raw `assistant` messages with `tool_calls` become ATIF `agent` steps whose
  `ToolCall` entries preserve the original CRMArena function names and JSON
  arguments.
- Raw `tool` messages become ATIF observations linked to the preceding tool call
  through `source_call_id` (the original `tool_call_id`).
- Task context is preserved: `task_id`, `task_type`, `gt_answer`, `reward`, and
  `agent_info` are carried into ATIF `extra`.
- `atif_to_std.py` applies ADP's shared tool-call normalization via
  `scripts/atif_to_std_common.py` while retaining the CRMArena-specific tools
  declared in `metadata.json` (including `respond`, which stays a custom tool
  rather than being rewritten to the built-in `finish`).

## Regenerating Samples

From the repository root. Pass the CRMArena result logs via `--source` (one file
per task category, or a glob). The committed sample keeps the first trajectory of
each task category, giving one representative per category:

```bash
export MY_DATASET=crmarena
export PYTHONPATH=`pwd`:$PYTHONPATH

# Extract raw data, keeping the first trajectory of each task category.
python datasets/$MY_DATASET/extract_raw.py --source /path/to/results_*_tool_call_*.json \
  | python -c "import sys,json
seen=set()
for line in sys.stdin:
    t=json.loads(line).get('task_type')
    if t not in seen:
        seen.add(t); sys.stdout.write(line)" \
  | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_raw.json

# Convert raw to ATIF
cat datasets/$MY_DATASET/sample_raw.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/raw_to_atif.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_atif.json

# Convert ATIF to standardized ATIF
cat datasets/$MY_DATASET/sample_atif.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/atif_to_std.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_std.json

# Convert to OpenHands v0 SFT
mkdir -p datasets/$MY_DATASET/sample_sft
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/openhands_v0/std_to_sft.py --is_web=no --api_env=execute_bash | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft/openhands_v0.json

# Convert to OpenHands SDK SFT
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/openhands_sdk/std_to_sft.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft/openhands_sdk.json
```

## Validate

```bash
python -m pytest tests/test_dataset_structure.py -v -k crmarena
python -m pytest tests/test_raw_schemas.py -v -k crmarena
python -m pytest tests/test_standardized_schemas.py -v -k crmarena
python -m pytest tests/test_std_to_sft_conversion.py -v -k crmarena
```

## Notes and Limitations

- Trajectories depend on the model used to generate the rollouts and on the
  state of the Salesforce sandbox at generation time; they are not deterministic
  across regenerations of the underlying rollouts.
- CRMArena exposes the same tool set to every task (with an optional
  SOQL/SOSL-enabled variant), so per-instance available-tool metadata is not
  populated; the universe of tools lives in `metadata.json`.
