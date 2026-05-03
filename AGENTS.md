# Agent Data Protocol - Repository Guidelines

This document captures key patterns and best practices for contributing to the Agent Data Protocol repository.

## Repository Structure

```
agent-data-protocol/
├── datasets/           # Dataset implementations
│   └── $DATASET_NAME/
│       ├── README.md
│       ├── extract_raw.py
│       ├── raw_to_standardized.py
│       ├── schema_raw.py
│       ├── api.py (required if ApiAction is used)
│       ├── sample_raw.json
│       ├── sample_std.json
│       ├── sample_sft.json
│       └── sample_sft/ (optional agent-specific copies)
│           └── sample_sft_$AGENT.json
├── agents/             # Agent-specific SFT converters
├── schema/             # ADP standardized format definitions
├── scripts/            # Utility scripts
└── tests/              # Validation tests
```

## Data Flow Pipeline

```
Raw Dataset      →  Standardized Format  →  Agent Specific SFT Format
     ↓                   ↓                       ↓
sample_raw.json  →  sample_std.json      →  sample_sft.json
```

## Key Requirements

### Dataset File Naming and Structure
- Every dataset directory must include `README.md`, `extract_raw.py`, `raw_to_standardized.py`, `schema_raw.py`, `sample_raw.json`, `sample_std.json`, and `sample_sft.json` unless there is a documented reason that the dataset is intentionally incomplete.
- If `sample_std.json` exists, root-level `sample_sft.json` is required. Optional agent-specific files may additionally live under `sample_sft/`, but they do not replace root-level `sample_sft.json`.
- Only these top-level JSON files are allowed in dataset directories:
  - `sample_raw.json`
  - `sample_std.json`
  - `sample_sft.json`
  - `generated_thoughts.json`
- Do not commit `full_raw.json`, `full_std.json`, `full_sft.json`, temporary chunks, downloaded corpora, scratch JSON, or alternate sample files such as `sample_fixed.json`.
- All JSON files MUST be valid JSON and MUST have a trailing newline.

### Generated Samples Must Come From the Pipeline
- Treat `sample_raw.json`, `sample_std.json`, and `sample_sft.json` as generated artifacts from the dataset scripts, not hand-edited fixtures.
- If a sample fails validation, fix `extract_raw.py`, `raw_to_standardized.py`, `schema_raw.py`, `api.py`, or the relevant agent converter, then regenerate the sample files.
- Do not directly patch sample JSON just to satisfy a failing test unless the same logic is also encoded in the generator that produced it.
- Keep the same records and order across `sample_raw.json`, `sample_std.json`, and `sample_sft.json`; the samples should represent the same tasks at each stage, with matching IDs between standardized and SFT files.
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

### Standardized Schema Components

**Actions:**
- `MessageAction`: Text-based assistant communication that is not a tool call or executable code.
- `CodeAction`: Code execution requests such as shell commands or notebook cells.
- `ApiAction`: API/function calls with `function` and `kwargs` fields.

**Observations:**
- `TextObservation`: Text-based responses with `source` set to a schema-allowed source (`user`, `agent`, or `environment`). Do not invent values like `system`, `os`, or `assistant`.
- `WebObservation`: Web page content.

### API and Schema Validity
- Every `ApiAction.function` used in `sample_std.json` must be implemented in that dataset's `api.py` with a compatible Python signature.
- Every `ApiAction.kwargs` object must validate by calling the function in `api.py`; include required parameters such as the `message` argument for `finish`.
- `schema_raw.py` must faithfully model the raw samples, and `sample_raw.json` must validate against it.
- Preserve the raw trajectory semantics when converting: do not drop repeated actions, consecutive tool calls, observations, failures, rewards, or terminal states unless the PR explains and justifies the filtering.
- Prefer shared agent converters in `agents/` when possible. Add a dataset-specific `std_to_sft.py` only when a real dataset-specific transformation is required, and explain why in the PR.

### Dataset Incorporation Do/Don't Checklist

**Do:**
- Read the dataset README/source card and cite the exact source, license, size, and split used.
- Map each raw role/action/observation to the closest ADP schema type before writing code.
- Keep extraction, standardization, and SFT conversion deterministic so future contributors can reproduce samples.
- Filter low-quality or unsuitable trajectories only with explicit code and an explanation in the PR.
- Run the focused dataset tests and fix the generator when tests reveal bad artifacts.
- Keep changes minimal and scoped to the dataset unless a shared schema or converter change is truly needed.

**Don't:**
- Do not add placeholder samples, unrelated trajectories, or samples that cannot be regenerated from the committed scripts.
- Do not manually change `from` roles, observation sources, or missing function parameters in JSON without fixing the converter.
- Do not leave failing pre-commit issues such as trailing whitespace, unsorted imports, invalid formatting, or missing EOF newlines.
- Do not add large raw downloads or full corpora to git; use ignored full files or streaming extraction instead.
- Do not add unneeded converter files, duplicate APIs, or bespoke logic when the existing shared code path works.
- Do not merge a dataset while promising to align it later; align it with current ADP conventions before review.

### PR Description Requirements for Dataset PRs
- The PR description must include the dataset source, license, size/split, files added, schema mapping summary, tests run, and any known limitations.
- Catalog every design decision that was unclear while implementing the dataset. For each decision, include:
  1. the question or ambiguity,
  2. the chosen approach,
  3. a concrete example from the dataset or code, and
  4. alternatives considered and why they were rejected.
- Example design-decision entry:
  - **Ambiguity:** Raw assistant messages sometimes contain shell commands embedded in prose.
  - **Chosen approach:** Convert only fenced/explicit command blocks to `CodeAction` and leave explanatory prose as `MessageAction`.
  - **Example:** `Run: pytest tests/test_api.py` becomes a `CodeAction`; `I will inspect the tests first` remains a `MessageAction`.
  - **Alternatives rejected:** Treating the whole assistant message as one `MessageAction` loses executable structure; converting all prose that mentions commands creates false tool calls.
- Include this catalog even when the decision seems small, such as how to handle system prompts, failed trajectories, missing final responses, unavailable tools, screenshots, rewards, or dataset-specific metadata.

## Commands

### Generate sample files
```bash
export MY_DATASET=your_dataset
export PYTHONPATH=`pwd`:$PYTHONPATH

# Extract raw data (5 samples)
python datasets/$MY_DATASET/extract_raw.py | head -5 | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_raw.json

# Convert the exact raw samples to standardized format
cat datasets/$MY_DATASET/sample_raw.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/raw_to_standardized.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_std.json

# Convert the exact standardized samples to the required root OpenHands SFT sample
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/openhands/std_to_sft.py --is_web=no --api_env=execute_bash | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft.json

# Optional: also save an agent-specific copy when maintaining sample_sft/ outputs
mkdir -p datasets/$MY_DATASET/sample_sft
cp datasets/$MY_DATASET/sample_sft.json datasets/$MY_DATASET/sample_sft/sample_sft_openhands.json
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
4. **Missing root `sample_sft.json`**: Required whenever `sample_std.json` exists.
5. **Hand-patched samples**: If a JSON fix is not reproducible by the scripts, reviewers should reject it.
6. **Mismatched sample stages**: `sample_raw`, `sample_std`, and `sample_sft` must correspond to the same records.
7. **Invalid observation sources**: Use schema-supported sources only: `user`, `agent`, and `environment`.
8. **Missing API parameters**: `ApiAction.kwargs` must satisfy the corresponding `api.py` function signature.
9. **Unneeded dataset-local converters**: Delete dataset-specific `std_to_sft.py` files unless they are required and justified.
10. **Large accidental commits**: Do not commit full corpora, generated chunks, screenshots, caches, or downloaded archives.

## Fix the Converter, Then Regenerate

If SFT conversion produces the wrong role for function calls, fix the conversion logic and regenerate `sample_sft.json`. The corrective logic belongs in the converter, not as a one-off edit to generated JSON:

```python
function_patterns = ["<function=", "<function_calls>", "<invoke name="]

if any(pattern in value for pattern in function_patterns):
    message["from"] = "function_call"
```

After changing a converter, regenerate samples with the commands above and rerun the focused validation tests.
