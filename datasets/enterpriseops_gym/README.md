# EnterpriseOps-Gym

## Description

EnterpriseOps-Gym (ServiceNow Research) is a containerized, resettable
enterprise simulation benchmark for evaluating LLM agents on **stateful,
multi-step planning and tool use** across realistic enterprise workflows. It
contains 1,150 expert-curated tasks over 8 domains — **Calendar, CSM, Drive,
Email, HR, ITSM, Teams, and Hybrid** — backed by 164 relational database tables
and ~512 MCP tools. Trajectories average ~9 steps (up to 34).

Unlike static, rollout-based datasets, EnterpriseOps-Gym is **evaluated by SQL
verifiers that check the final environment state**, not by matching an action
sequence. Each published task carries a system prompt, a user request, the
oracle tool set, the MCP server configuration, and the SQL verifiers.

ADP stores each agent rollout as an OpenAI-style chat trajectory
(`system` / `user` / `assistant` / `tool` messages) where assistant turns carry
MCP tool calls and `tool` turns carry the environment observations. This is the
same raw shape used by the `crmarena` and `enterpriselab` datasets, so
EnterpriseOps-Gym reuses ADP's shared `raw_to_atif` / `atif_to_std` converters.

## Dataset Information

- **Original Paper**: [EnterpriseOps-Gym: Environments and Evaluations for Stateful Agentic Planning and Tool Use in Enterprise Settings](https://arxiv.org/abs/2603.13594)
- **Original Repository**: https://github.com/ServiceNow/EnterpriseOps-Gym
- **Source Data**: https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym
  (public split ≈ 60% of tasks; configs `oracle`, `plus_5_tools`,
  `plus_10_tools`, `plus_15_tools`; one split per domain)
- **Related ADP paper**: https://arxiv.org/abs/2510.24702
- **License**: Apache 2.0 (see [`LICENSE`](LICENSE) and the upstream sources).
- **Domains (8)**: `calendar`, `csm`, `drive`, `email`, `hr`, `itsm`, `teams`,
  `hybrid`.
- **Tools**: declared in [`metadata.json`](metadata.json). Names are the real
  EnterpriseOps-Gym MCP tools (taken from each task's `selected_tools`);
  parameter schemas are inferred from the sample tool calls because the full
  tool schemas live inside the containerized MCP servers, not the public repo.

## Trajectories are generated, not downloaded

**Important:** The Hugging Face dataset publishes *tasks + SQL verifiers*, **not**
agent trajectories. A trajectory only exists once an agent is run against the
live, containerized gym MCP servers. Use
[`generate_trajectories.py`](generate_trajectories.py):

1. **`--from-hf`** (default, no gym/containers needed) — download the published
   tasks and build *illustrative reference trajectories* grounded in the real
   task fields:
   ```bash
   export PYTHONPATH=`pwd`:$PYTHONPATH
   python datasets/enterpriseops_gym/generate_trajectories.py --from-hf \
       --config oracle --per-domain 1 \
       --output datasets/enterpriseops_gym/enterpriseops_gym_gold.json
   ```
   (Needs the commented extras in `requirements.txt`: `datasets`,
   `huggingface_hub`, `pandas`, `pyarrow`.)

2. **`--from-results`** — convert result logs from EnterpriseOps-Gym's own
   `evaluate.py` into authentic gold trajectories:
   ```bash
   python datasets/enterpriseops_gym/generate_trajectories.py \
       --from-results /path/to/EnterpriseOps-Gym/results \
       --output datasets/enterpriseops_gym/enterpriseops_gym_gold.json
   ```

3. **`--run`** — placeholder for a live driver against the gym MCP servers
   (requires the containerized environment + model credentials).

`enterpriseops_gym_gold.json` is a local build artifact (git-ignored). The
committed `sample_raw.json` is used as the fallback source when it is absent.

### About the committed `sample_raw.json`

Because authentic trajectories require running the containerized gym, the five
committed samples are **illustrative reference trajectories**: the task id,
domain, system/user prompts, oracle tool list, and SQL verifiers are the real
published EnterpriseOps-Gym task, while the tool-call arguments and observations
are compact, representative values that demonstrate the trajectory shape. Each
record carries a `note` field. Example access tokens in `gym_servers_config`
are redacted. Regenerate with `--from-results` + `make_samples.py` to replace
the illustrative steps with authentic ones.

## Regenerating the committed samples

```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
python datasets/enterpriseops_gym/make_samples.py --num-samples 5
```

`make_samples.py` reads `enterpriseops_gym_gold.json` when present, otherwise
falls back to the committed `sample_raw.json`. It writes `metadata.json`,
`sample_raw.json`, `sample_atif.json`, `sample_std.json`, and
`sample_sft/{openhands_sdk,sweagent}.json`.

## Schema Mapping

- Raw `system` / `user` / `assistant` messages are preserved as ATIF steps.
- Raw assistant `tool_calls` become ATIF `ToolCall` entries with the original
  MCP tool names and JSON arguments.
- Raw `tool` messages become ATIF observations, linked to the preceding
  assistant tool call.
- `atif_to_std.py` applies ADP's shared tool-call normalization while retaining
  EnterpriseOps-Gym-specific tools declared in `metadata.json`.
- `sample_sft/openhands_sdk.json` is generated from the standardized ATIF sample
  through the shared OpenHands SDK converter.

Building samples (raw → ATIF → std → SFT) needs `pydantic>=2.12` and the
OpenHands SDK. Pulling tasks from Hugging Face additionally needs `datasets`,
`huggingface_hub`, `pandas`, and `pyarrow` (see the commented extras in
`requirements.txt`).

Diagnose OpenHands import problems with:

```bash
python datasets/enterpriseops_gym/make_samples.py --check-openhands
```

## Validation

```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
python -m pytest tests/test_dataset_structure.py -v -k enterpriseops_gym
python -m pytest tests/test_raw_schemas.py -v -k enterpriseops_gym
python -m pytest tests/test_standardized_schemas.py -v -k enterpriseops_gym
```

## Citation

Please cite EnterpriseOps-Gym using the citation requested by the upstream
authors:

```bibtex
@article{enterpriseopsgym2026,
  title={EnterpriseOps-Gym: Environments and Evaluations for Stateful Agentic
         Planning and Tool Use in Enterprise Settings},
  author={ServiceNow Research},
  journal={arXiv preprint arXiv:2603.13594},
  year={2026}
}
```
