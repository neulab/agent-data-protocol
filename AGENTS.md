# Agent Data Protocol - Repository Guidelines

This document captures key patterns and best practices for contributing to the Agent Data Protocol repository.

## Repository Structure

```
agent-data-protocol/
├── datasets/           # Dataset implementations
│   └── $DATASET_NAME/
│       ├── README.md
│       ├── extract_raw.py
│       ├── raw_to_atif.py
│       ├── atif_to_std.py
│       ├── raw_to_standardized.py
│       ├── metadata.json
│       ├── schema_raw.py
│       ├── sample_raw.json
│       ├── sample_atif.json
│       ├── sample_std.json
│       └── sample_sft/
│           ├── openhands_v0.json
│           └── $AGENT_NAME.json
├── agents/             # Agent-specific SFT converters
├── schema/             # ATIF and ADP schema definitions
├── scripts/            # Utility scripts
└── tests/              # Validation tests
```

## Data Flow Pipeline

```
sample_raw.json → raw_to_atif.py → sample_atif.json → atif_to_std.py → sample_std.json → agents/*/std_to_sft.py → sample_sft/<agent_name>.json
```

## Key Requirements

### Dataset File Naming and Structure
- Every dataset directory must include `README.md`, `extract_raw.py`, `raw_to_atif.py`, `atif_to_std.py`, `raw_to_standardized.py`, `schema_raw.py`, `sample_raw.json`, `sample_atif.json`, `sample_std.json`, and `sample_sft/openhands_v0.json` unless there is a documented reason that the dataset is intentionally incomplete.
- If `sample_std.json` exists, `sample_sft/openhands_v0.json` is required. Additional agent-specific files may live under `sample_sft/` using the exact agent identifier as the filename, such as `sample_sft/sweagent.json`.
- Only these top-level JSON files are allowed in dataset directories:
  - `sample_raw.json`
  - `sample_atif.json`
  - `sample_std.json`
  - `generated_thoughts.json`
- Do not commit `full_raw.json`, `full_atif.json`, `full_std.json`, `full_sft.json`, `full_raw.jsonl`, `full_atif.jsonl`, `full_std.jsonl`, `full_sft.jsonl`, temporary chunks, downloaded corpora, scratch JSON, or alternate sample files such as `sample_fixed.json`.
- All JSON files MUST be valid JSON and MUST have a trailing newline.

### Generated Samples Must Come From the Pipeline
- Treat `sample_raw.json`, `sample_atif.json`, `sample_std.json`, and files under `sample_sft/` as generated artifacts from the dataset scripts, not hand-edited fixtures.
- If a sample fails validation, fix `extract_raw.py`, `raw_to_atif.py`, `atif_to_std.py`, `raw_to_standardized.py`, `schema_raw.py`, `metadata.json`, or the relevant shared agent converter, then regenerate the sample files.
- Put dataset-specific normalization in `raw_to_atif.py` or `atif_to_std.py`. Do not add dataset-local `std_to_sft.py` files.
- Do not directly patch sample JSON just to satisfy a failing test unless the same logic is also encoded in the generator that produced it.
- Keep the same records and order across `sample_raw.json`, `sample_atif.json`, `sample_std.json`, and each `sample_sft/<agent_name>.json`; the samples should represent the same tasks at each stage, with matching IDs between standardized and SFT files.
- Use small representative samples, normally 3-5 trajectories, that include important edge cases such as tool calls, command output, final answers, and any dataset-specific action types.

### SFT Format Requirements

**Critical**: Messages containing function call patterns MUST use `"from": "function_call"`, not `"from": "gpt"`, `"human"`, or any other role.

Function call patterns that trigger this requirement:
- `<function=`
- `<function_calls>`
- `<invoke name=`

Example correct format:
```json
{
  "from": "function_call",
  "value": "I'll run the command.\n\n<function=execute_bash>\n<parameter=command>ls -la</parameter>\n</function>"
}
```

### ATIF Standardized Schema Components

Committed `sample_atif.json` and `sample_std.json` files are ATIF trajectories. `raw_to_atif.py` should preserve raw trajectory structure with minimal formatting changes. `atif_to_std.py` should keep ATIF in and ATIF out while standardizing tool-call names and arguments for downstream converters.

**Steps:**
- `source`: ATIF step source, one of `system`, `user`, or `agent`.
- `message`: The natural-language content for the step.
- `tool_calls`: Agent tool calls with `function_name`, `arguments`, and `tool_call_id`.
- `observation`: Tool/environment results linked to tool calls through `source_call_id`.

**Versioning:**
- The canonical ATIF schema version lives in `schema/atif.py` as `ATIF_SCHEMA_VERSION`.
- `ATIFTrajectory` includes a root-level `schema_version`; committed `sample_atif.json` and `sample_std.json` files must include the current value explicitly.
- Any schema-impacting Python change under `schema/` must bump `SCHEMA_VERSION`; CI checks this with `scripts/check_schema_version_bump.py`.

### Tool and Schema Validity
- Every `ToolCall.function_name` used in `sample_std.json` must be a built-in standardized tool, a browser action, or a custom tool declared in `metadata.json`.
- Every `ToolCall.arguments` object must match the corresponding tool schema in `metadata.json` or the built-in standardized tool signature.
- If a trajectory has per-instance tool availability, keep it as ATIF metadata such as `extra.adp_available_apis`; every used custom tool in that trajectory must appear in the available set.
- Only populate per-instance available tools when the source data explicitly specifies them. Do not fill it with all functions from `metadata.json`, and do not infer it merely from the tools used in the trajectory.
- `schema_raw.py` must faithfully model the raw samples, and `sample_raw.json` must validate against it.
- Preserve the raw trajectory semantics when converting: do not drop repeated actions, consecutive tool calls, observations, failures, rewards, or terminal states unless the PR explains and justifies the filtering.
- Use shared agent converters in `agents/`. Dataset-local `std_to_sft.py` files are not allowed; move dataset-specific normalization into `raw_to_atif.py` or `atif_to_std.py`.

### Dataset Incorporation Do/Don't Checklist

**Do:**
- Read the dataset README/source card and cite the exact source, license, size, and split used.
- Map each raw role/action/observation to the closest ATIF step, tool call, or observation result before writing code.
- Keep extraction, standardization, and SFT conversion deterministic so future contributors can reproduce samples.
- Filter low-quality or unsuitable trajectories only with explicit code and an explanation in the PR.
- Run the focused dataset tests and fix the generator when tests reveal bad artifacts.
- Keep changes minimal and scoped to the dataset unless a shared schema or converter change is truly needed.

**Don't:**
- Do not add placeholder samples, unrelated trajectories, or samples that cannot be regenerated from the committed scripts.
- Do not manually change `from` roles, observation sources, or missing function parameters in JSON without fixing the converter.
- Do not leave failing pre-commit issues such as trailing whitespace, unsorted imports, invalid formatting, or missing EOF newlines.
- Do not add large raw downloads or full corpora to git; use ignored full files or streaming extraction instead.
- Do not add dataset-local SFT converters, duplicate APIs, or bespoke logic when the existing shared code path works.
- Do not merge a dataset while promising to align it later; align it with current ATIF std conventions before review.

### PR Description Requirements for Dataset PRs
- The PR description must include the dataset source, license, size/split, files added, schema mapping summary, tests run, and any known limitations.
- Catalog every design decision that was unclear while implementing the dataset. For each decision, include:
  1. the question or ambiguity,
  2. the chosen approach,
  3. a concrete example from the dataset or code, and
  4. alternatives considered and why they were rejected.
- Example design-decision entry:
  - **Ambiguity:** Raw assistant messages sometimes contain shell commands embedded in prose.
  - **Chosen approach:** Convert only fenced/explicit command blocks to tool calls and leave explanatory prose in the ATIF step message.
  - **Example:** `Run: pytest tests/test_api.py` becomes an `execute_bash` tool call; `I will inspect the tests first` remains message text.
  - **Alternatives rejected:** Treating the whole assistant message as plain ATIF message text loses executable structure; converting all prose that mentions commands creates false tool calls.
- Include this catalog even when the decision seems small, such as how to handle system prompts, failed trajectories, missing final responses, unavailable tools, screenshots, rewards, or dataset-specific metadata.

## Commands

### Generate sample files
```bash
export MY_DATASET=your_dataset
export PYTHONPATH=`pwd`:$PYTHONPATH

# Extract raw data (5 samples)
python datasets/$MY_DATASET/extract_raw.py | head -5 | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_raw.json

# Convert the exact raw samples to ATIF format
cat datasets/$MY_DATASET/sample_raw.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/raw_to_atif.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_atif.json

# Normalize the exact ATIF samples to ATIF std format
cat datasets/$MY_DATASET/sample_atif.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/atif_to_std.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_std.json

# Convert the exact ATIF std samples to the required OpenHands v0 SFT sample
mkdir -p datasets/$MY_DATASET/sample_sft
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/openhands_v0/std_to_sft.py --is_web=no --api_env=execute_bash | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft/openhands_v0.json
```

### Run tests
```bash
# All tests
python -m pytest tests/ -v

# Tests for a specific dataset
python -m pytest tests/ -v -k "dataset_name"

# Key validation tests for dataset PRs
python -m pytest tests/test_dataset_structure.py -v
python -m pytest tests/test_raw_schemas.py -v -k "dataset_name"
python -m pytest tests/test_standardized_schemas.py -v -k "dataset_name"
python -m pytest tests/test_std_to_sft_conversion.py -v -k "dataset_name"
python -m pytest tests/test_datasets_from_parameter.py -v
```

## Common Issues Learned From Prior PRs

1. **Missing trailing newline**: All JSON and Python files must end with `\n`.
2. **Wrong `from` field**: SFT messages containing function-call syntax must use `"from": "function_call"`.
3. **Extra JSON files**: Remove temporary or alternate `.json` files before committing.
4. **Missing `sample_sft/openhands_v0.json`**: Required whenever `sample_std.json` exists.
5. **Hand-patched samples**: If a JSON fix is not reproducible by the scripts, reviewers should reject it.
6. **Mismatched sample stages**: `sample_raw`, `sample_atif`, `sample_std`, and `sample_sft/<agent_name>` must correspond to the same records.
7. **Invalid step sources**: Use ATIF-supported step sources only: `system`, `user`, and `agent`.
8. **Missing tool parameters**: `ToolCall.arguments` must satisfy the corresponding standardized or custom tool signature.
9. **Dataset-local SFT converters**: Do not add dataset-specific `std_to_sft.py` files. Put normalization in `raw_to_atif.py` or `atif_to_std.py`.
10. **Large accidental commits**: Do not commit full corpora, generated chunks, screenshots, caches, or downloaded archives.

## Fix the Converter, Then Regenerate

If SFT conversion produces the wrong role for function calls, fix the conversion logic and regenerate the affected `sample_sft/<agent_name>.json`. The corrective logic belongs in the converter, not as a one-off edit to generated JSON:

```python
function_patterns = ["<function=", "<function_calls>", "<invoke name="]

if any(pattern in value for pattern in function_patterns):
    message["from"] = "function_call"
```

After changing a converter, regenerate samples with the commands above and rerun the focused validation tests.
