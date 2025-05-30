# Validator Scripts

This directory contains scripts for validating and analyzing the quality of datasets.

## SFT Quality Control

The `sft_quality_control.py` script analyzes SFT data quality and generates metrics and visualizations.

### Step 1: Generate Sample SFT Files

Before running quality control, ensure you have sample_sft.json files for your datasets:

```bash
export MY_DATASET=dataset_name
export PYTHONPATH=`pwd`:$PYTHONPATH

# Determine if this is a web-based dataset
IS_WEB="no"
if [[ "$MY_DATASET" == "mind2web" || "$MY_DATASET" == "synatra" || "$MY_DATASET" == "webarena_successful" || "$MY_DATASET" == "weblinx" ]]; then
    IS_WEB="yes"
fi

# Convert sample.json to sample_sft.json
cat datasets/$MY_DATASET/sample.json | python scripts/json_to_jsonl.py | python scripts/std_to_sft.py --is_web=$IS_WEB --chunk=all | python scripts/jsonl_to_indented_json.py > datasets/$MY_DATASET/sample_sft.json
```

### Step 2: Run Quality Control Analysis

```bash
# Analyze all datasets and save results to the output directory
python validator/sft_quality_control.py --input_dirs datasets --output_dir quality_control_results

# Analyze a specific dataset
python validator/sft_quality_control.py --input_dirs datasets/dataset_name --output_dir quality_control_results
```

### Step 3: Review Quality Control Results

The script generates several visualizations and a CSV file with metrics:

- `function_calls_per_turn.png`: Distribution of function calls per turn across datasets
- `function_calls_without_thoughts.png`: Percentage of function calls without preceding thoughts
- `role_turns_per_conversation.png`: Distribution of role turns per conversation
- `sft_quality_metrics.csv`: Detailed metrics for each dataset

These metrics help identify datasets with missing function calls, low thought usage, or other quality issues.