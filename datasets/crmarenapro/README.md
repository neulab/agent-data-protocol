# CRMArena-Pro

## Description

CRMArena-Pro is the expanded version of Salesforce AI Research's CRMArena
benchmark for evaluating LLM agents on professional Customer Relationship
Management (CRM) tasks in realistic Salesforce environments. It extends the
original CRMArena with **two business orgs — B2B and B2C**, additional
**sales** task categories (lead routing/qualification, quote approval, sales
analytics, etc.), **confidentiality-awareness** tasks (recognising and refusing
requests for private/internal/confidential data), and **interactive multi-turn**
scenarios in which the agent talks to a simulated user.

Like CRMArena, the public `Salesforce/CRMArenaPro` dataset ships **tasks and
ground-truth answers**, not agent trajectories. This ADP integration therefore
uses **agent rollouts**: the CRMArena ReAct agent (`--agent_strategy react`) is
run over the tasks against the B2B / B2C Salesforce sandboxes, and each
rollout's full message history (system prompt → user query → assistant
`<thought>`/`<execute>`/`<respond>` turns → Salesforce observations) is converted
into an ATIF trajectory.

Unlike the original CRMArena (which uses the OpenAI tool-calling `tool_call`
agent), CRMArena-Pro's B2B/B2C orgs are only supported by the **ReAct
`ChatAgent`** (the upstream `tool_call` strategy is restricted to the `original`
org). The agent therefore acts through free-text ReAct actions rather than
native function calls:

- `<execute> ... </execute>` — a SOQL/SOSL query (or expression) run against the
  Salesforce instance; the result comes back as an observation prefixed
  `Salesforce instance output:`.
- `<respond> ... </respond>` — the agent's answer to the user. In single-turn
  tasks this submits the final answer; in interactive tasks it sends a message
  to the simulated user, who may reply.

The dataset focuses on:
- CRM workflow, analytics, sales, and confidentiality tasks across the B2B and
  B2C CRMArena-Pro orgs
- Multi-turn Salesforce SOQL/SOSL tool use with real query errors and agent
  recovery/retries
- Both successful and unsuccessful task trajectories
- Single-turn and (optionally) interactive multi-turn rollouts, which convert
  through the same pipeline

The committed sample contains one representative trajectory per task category
for **both** orgs (b2b and b2c), spanning service (`case_routing`), knowledge
(`knowledge_qa`), analytics (`monthly_trend_analysis`), sales (`lead_routing`,
`quote_approval`), and confidentiality (`private_customer_information`) tasks.

## Dataset Information

- Original paper: [CRMArena-Pro: Holistic Assessment of LLM Agents Across Diverse Business Scenarios and Interactions](https://arxiv.org/abs/2505.18878)
- Original dataset: https://huggingface.co/datasets/Salesforce/CRMArenaPro
- Original repository: https://github.com/SalesforceAIResearch/CRMArena
- Related ADP paper: https://arxiv.org/abs/2510.24702
- License: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0) — see [LICENSE](LICENSE)
- Orgs used: `b2b` and `b2c` (CRMArena-Pro); rollouts over the task set for each org.
- Generation model: the rollouts committed here were produced with GPT-5.1 via an
  OpenAI-compatible endpoint (any tool-calling/instruction-following model works;
  trajectories are not deterministic across regenerations).
- Raw source used locally: CRMArena-Pro ReAct result logs
  (`results/.../results_<model>_react_<category>.json`) produced by the CRMArena
  ReAct agent; each record's `traj` is the ReAct message list.
- Regeneration deps for producing rollouts are listed in
  [`requirements.txt`](requirements.txt); the ADP conversion scripts themselves need
  only the standard library and shared repo dependencies.

## Tools

CRMArena-Pro's ReAct action space maps onto ADP tools as follows:

- `execute` (a `<execute>` SOQL/SOSL/expression block) is preserved in the raw
  ATIF as an `execute` tool call and then normalized by the shared
  ATIF-to-standardized step (`normalize_atif_trajectory`) to the built-in code
  execution tool (`execute_ipython_cell`, standardized to `python`). `python` is
  therefore declared through `code_enabled` in [`metadata.json`](metadata.json).
- `respond` is declared as a `custom_tool` in [`metadata.json`](metadata.json)
  (it stays a custom tool rather than being rewritten to the built-in `finish`,
  matching the original CRMArena integration), because it also carries the
  agent's interim messages to the user in interactive tasks.

## Schema Mapping

- Each rollout record's `traj` (renamed to `messages` by `extract_raw.py`) is a
  ReAct message list and is converted by this dataset's `raw_to_atif.py`.
- Raw `system` / first `user` messages become ATIF `system` / `user` steps.
  (Some reasoning models merge the system prompt into the first user message;
  it is still mapped to a `user` step.)
- Assistant messages become ATIF `agent` steps: the `<thought>` text becomes the
  step message and the `<execute>` or `<respond>` action becomes a `ToolCall`.
- `user` messages prefixed `Salesforce instance output:` become the ATIF
  observation linked to the preceding `execute` tool call through
  `source_call_id`; any other `user` message (an interactive user reply) becomes
  a `user` step.
- Task context is preserved: `task_id`, `task_type`, `gt_answer`, `reward`,
  `org_type`, and `interactive` are carried into ATIF `extra`.
- `atif_to_std.py` applies ADP's shared tool-call normalization via
  `scripts/atif_to_std_common.py` while retaining the CRMArena-Pro-specific
  `respond` custom tool declared in `metadata.json`.

## Regenerating Samples

Rollouts are produced with the CRMArena repository
(https://github.com/SalesforceAIResearch/CRMArena). Run its ReAct agent against
the B2B / B2C Salesforce sandboxes (credentials and the LLM endpoint are read
from CRMArena's `.env`), e.g. per task category and org:

```bash
# In the CRMArena repo, for each org (b2b, b2c) and task category:
python run_tasks.py \
  --model gpt-5.1 --agent_strategy react --agent_eval_mode aided \
  --llm_provider openai --org_type b2b \
  --task_category case_routing \
  --log_dir results/adp_b2b
# (add --interactive for multi-turn rollouts)
```

Then, from the agent-data-protocol repository root, convert the result logs.
Pass the CRMArena-Pro result logs via `--source` (b2b and b2c files together;
`extract_raw.py` infers the org and interactive flag from the file path). The
committed sample keeps the first trajectory of each (org, task category):

```bash
export MY_DATASET=crmarenapro
export PYTHONPATH=`pwd`:$PYTHONPATH

# Extract raw data, keeping the first trajectory of each (org, task category).
python datasets/$MY_DATASET/extract_raw.py --source /path/to/results_*_react_*.json \
  | python -c "import sys,json
seen=set()
for line in sys.stdin:
    r=json.loads(line); k=(r.get('org_type'), r.get('task_type'))
    if k not in seen:
        seen.add(k); sys.stdout.write(line)" \
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
python -m pytest tests/test_dataset_structure.py -v -k crmarenapro
python -m pytest tests/test_raw_schemas.py -v -k crmarenapro
python -m pytest tests/test_standardized_schemas.py -v -k crmarenapro
python -m pytest tests/test_std_to_sft_conversion.py -v -k crmarenapro
```

## Notes and Limitations

- Trajectories depend on the model used to generate the rollouts and on the
  state of the B2B / B2C Salesforce sandboxes at generation time; they are not
  deterministic across regenerations of the underlying rollouts.
- CRMArena-Pro exposes the same ReAct action space (`execute`, `respond`) to
  every task, so per-instance available-tool metadata is not populated; the
  universe of tools lives in `metadata.json` plus the built-in `python`
  (code execution) tool.
- Interactive multi-turn rollouts (`--interactive`) convert through the same
  pipeline: simulated user replies become `user` steps interleaved with the
  agent's `respond`/`execute` steps.
