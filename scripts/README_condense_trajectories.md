# Trajectory Condensation Utility

This utility applies OpenHands SDK context condensation to SFT trajectories, splitting them when condensation occurs.

## Overview

The `condense_trajectories.py` script uses the OpenHands agent SDK's context condenser to apply condensation to trajectories stored in the SFT format. When condensation occurs (due to the conversation history exceeding a threshold), the trajectory is split into multiple segments since the prefix (system prompt) changes after condensation.

## Usage

```bash
python scripts/condense_trajectories.py \
  <input_file> \
  <output_file> \
  [--max-size MAX_SIZE] \
  [--keep-first KEEP_FIRST] \
  [--use-mock-condenser] \
  [--llm-model MODEL] \
  [--llm-base-url URL]
```

### Arguments

- `input_file`: Path to input sample_sft_openhands.json file
- `output_file`: Path to output condensed trajectories file
- `--max-size`: Maximum number of events before condensation (default: 10)
- `--keep-first`: Number of initial events to always keep (default: 2)
- `--use-mock-condenser`: Use mock condenser instead of LLM-based condenser (for testing)
- `--llm-model`: LLM model to use (default: from LLM_MODEL env var)
- `--llm-base-url`: LLM base URL (default: from LLM_BASE_URL env var)

### Environment Variables

When using the LLM-based condenser (without `--use-mock-condenser`):
- `LLM_API_KEY`: Required. API key for the LLM service.
- `LLM_MODEL`: Optional. Model identifier (default: anthropic/claude-3-5-sonnet-20241022)
- `LLM_BASE_URL`: Optional. Base URL for the LLM API

## Examples

### Using Mock Condenser (for testing)

```bash
python scripts/condense_trajectories.py \
  datasets/swe-smith/sample_sft/sample_sft_openhands.json \
  output_condensed.json \
  --max-size 12 \
  --keep-first 2 \
  --use-mock-condenser
```

### Using LLM Condenser

```bash
export LLM_API_KEY="your-api-key"
export LLM_MODEL="anthropic/claude-3-5-sonnet-20241022"

python scripts/condense_trajectories.py \
  datasets/swe-smith/sample_sft/sample_sft_openhands.json \
  output_condensed.json \
  --max-size 120 \
  --keep-first 4
```

## Input/Output Format

### Input
- Format: `sample_sft_openhands.json` (list of trajectory objects)
- Each trajectory has: `id`, `system`, `conversations`
- N total trajectories

### Output
- Same format as input
- N*M trajectories, where M is the average number of condensations + 1
- Each segment gets a unique ID: `{original_id}_seg{index}`

## How It Works

1. **Load trajectories**: Reads the input JSON file containing SFT trajectories
2. **Convert to events**: Transforms each conversation turn into a MessageEvent
3. **Apply condenser**: Iteratively builds a View and applies condensation when threshold is exceeded
4. **Split on condensation**: When condensation occurs, creates a new trajectory segment
5. **Track condensation events**: Maintains a list of all events including Condensation events
6. **Output segments**: Writes all trajectory segments to the output file

## Condensation Details

When a trajectory exceeds the `max-size` threshold:
1. The condenser identifies events to forget (those not in the first `keep_first` or last few events)
2. Creates a summary of the forgotten events (LLM-generated or mock)
3. A Condensation event is added to the history
4. The trajectory is split at this point
5. Subsequent segments start with the condensed view

## Testing

The script includes comprehensive logging to track:
- Trajectory processing progress
- Condensation triggers and details
- Segment creation
- Final statistics

Use `jq` to inspect the output:

```bash
# Count trajectories
jq 'length' output_condensed.json

# View trajectory IDs and conversation counts
jq '[.[] | {id, conversations: (.conversations | length)}]' output_condensed.json

# Inspect a specific trajectory
jq '.[0]' output_condensed.json
```

## Dependencies

- `openhands-sdk`: For context condenser and event handling
- `pydantic`: For data validation
- Standard library: `json`, `logging`, `argparse`, `os`, `sys`

Install the OpenHands SDK:

```bash
cd /path/to/software-agent-sdk
pip install -e ./openhands-sdk
```
