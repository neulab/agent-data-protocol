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
│       ├── schema_raw.py (optional)
│       ├── api.py (optional)
│       ├── sample_raw.json
│       ├── sample_std.json
│       ├── sample_sft.json
│       └── sample_sft/
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

### File Naming
- Only these JSON files are allowed in dataset directories:
  - `sample_raw.json`
  - `sample_std.json`
  - `sample_sft.json`
  - `generated_thoughts.json`
- All JSON files MUST have a trailing newline

### SFT Format Requirements

**Critical**: Messages containing function call patterns MUST use `"from": "function_call"`, not `"from": "gpt"`.

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
- `MessageAction`: Text-based communication
- `CodeAction`: Code execution requests
- `ApiAction`: API/function calls with `function` and `kwargs` fields

**Observations:**
- `TextObservation`: Text-based responses with `source` field (user/environment)
- `WebObservation`: Web page content

## Commands

### Generate sample files
```bash
export MY_DATASET=your_dataset
export PYTHONPATH=`pwd`:$PYTHONPATH

# Extract raw data (5 samples)
python datasets/$MY_DATASET/extract_raw.py | head -5 | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_raw.json

# Convert to standardized format
cat datasets/$MY_DATASET/sample_raw.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/raw_to_standardized.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_std.json

# Convert to SFT format (OpenHands)
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/openhands/std_to_sft.py --is_web=no --api_env=execute_bash | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft/sample_sft_openhands.json
```

### Run tests
```bash
# All tests
python -m pytest tests/ -v

# Tests for specific dataset
python -m pytest tests/ -v -k "dataset_name"

# Key validation tests
python -m pytest tests/test_dataset_structure.py -v
python -m pytest tests/test_datasets_from_parameter.py -v
python -m pytest tests/test_standardized_schemas.py -v
```

## Common Issues

1. **Missing trailing newline**: All JSON files must end with `\n`
2. **Wrong `from` field**: Function calls must use `"from": "function_call"`
3. **Extra JSON files**: Remove any temporary `.json` files before committing
4. **Missing `sample_sft.json`**: Required at root level if `sample_std.json` exists

## Post-Processing SFT Files

If your SFT conversion produces `"from": "gpt"` for function calls, apply this fix:

```python
import json

function_patterns = ['<function=', '<function_calls>', '<invoke name=']

with open('sample_sft.json', 'r') as f:
    data = json.load(f)

for item in data:
    for message in item.get('conversations', []):
        value = message.get('value', '')
        if any(p in value for p in function_patterns):
            if message.get('from') == 'gpt':
                message['from'] = 'function_call'

with open('sample_sft.json', 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
```
