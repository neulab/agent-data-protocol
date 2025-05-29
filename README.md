# agent-data-collection

This is a repository for agent training data collection by CMU, OSU, and HKU.

- `datasets/`: Contains datasets, each with at least the following elements
  - `README.md`: A description of the dataset
  - `sample_raw.json`: 2-5 raw samples from the corpus in the original format
  - `sample_standardized.json`: 2-5 samples from the corpus in our standardized format
  - `extract_raw.py`: A script that extracts a raw jsonl file from the corpus
  - `raw_to_standardized.py`: A script that converts the raw jsonl file to our standardized format jsonl
- `validator/`: Contains scripts that validate the dataset format
- `scripts/`:
  - `jsonl_to_indented_json.py`: Converts a jsonl file to an indented json file for easier viewing
  - `json_to_jsonl.py`: Converts a JSON file to a JSONL file
  - `sft_quality_control.py`: Analyzes SFT data quality and generates metrics and visualizations
  - `std_to_sft.py`: Converts standardized data to SFT format

## Adding a new dataset

### Step 1: Create Sample Data

To add a new dataset, the first step is to create sample data in order `extract_raw.py`, which will
output a jsonl file containing the raw data. You can view
[datasets/mind2web/extract_raw.py](datasets/mind2web/extract_raw.py) for an example.

Once you have created this, run the following command to create a sample (ignore the BrokenPipeError):

```bash
export MY_DATASET=dataset_name
python datasets/$MY_DATASET/extract_raw.py | head -n 3 | python scripts/jsonl_to_indented_json.py > datasets/$MY_DATASET/sample_raw.json
```

This sample data will form the basis of our discussion regarding the standardized dataset format.

### Step 2: Write Convertor to Standardized Format

Once we have our standardized format (not yet), we will create a script that converts, line-by-line, a jsonl file in the raw format to one in the standardized format in `raw_to_standardized.py`.

We can then apply this to the sample data to create a sample in the standardized format.

```bash
export MY_DATASET=dataset_name
export PYTHONPATH=`pwd`:$PYTHONPATH
cat datasets/$MY_DATASET/sample_raw.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/raw_to_standardized.py | python scripts/jsonl_to_indented_json.py > datasets/$MY_DATASET/sample.json
```

Run the validator script on the dataset to ensure that it is in the correct format:

```bash
pytest tests/test_curated_schemas.py
```

### Step 3: Write README

Write a README.md file in the dataset directory that describes the dataset, including the source, the format, and any other relevant information.

## Downloading and Converting Full Data to SFT Format

We prefer to use `.jsonl` files for downloading the full datasets

### Step 1: Download Full Raw Data

```bash
export MY_DATASET=dataset_name
python datasets/$MY_DATASET/extract_raw.py > datasets/$MY_DATASET/full_raw.jsonl
```

### Step 2: Convert Raw Data to the Standardized Schema

```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
cat datasets/$MY_DATASET/full_raw.jsonl | python datasets/$MY_DATASET/raw_to_standardized.py > datasets/$MY_DATASET/full_std.jsonl
```
### Step 3: Convert Standardized Data to SFT Format

```bash
cat datasets/$MY_DATASET/full_std.jsonl | python -u scripts/std_to_sft.py --is_web=no --chunk=all > datasets/$MY_DATASET/full_sft.jsonl
```
Use `--is_web=yes` for web only based datasets like `mind2web, synatra`

## Running Quality Control Analysis

The repository includes a script to analyze the quality of SFT data and generate metrics and visualizations. This is useful for understanding the distribution of function calls, thought usage, and other quality metrics across datasets.

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
python scripts/sft_quality_control.py --input_dirs datasets --output_dir quality_control_results

# Analyze a specific dataset
python scripts/sft_quality_control.py --input_dirs datasets/dataset_name --output_dir quality_control_results
```

### Step 3: Review Quality Control Results

The script generates several visualizations and a CSV file with metrics:

- `function_calls_per_turn.png`: Distribution of function calls per turn across datasets
- `function_calls_without_thoughts.png`: Percentage of function calls without preceding thoughts
- `role_turns_per_conversation.png`: Distribution of role turns per conversation
- `sft_quality_metrics.csv`: Detailed metrics for each dataset

These metrics help identify datasets with missing function calls, low thought usage, or other quality issues.

# Run quality control on specific datasets
python scripts/sft_quality_control.py --input_dirs datasets/mind2web datasets/synatra --output_dir quality_control_results
```

This will generate:
- CSV file with quality metrics for each dataset
- Visualizations of function calls per turn
- Visualizations of function calls without thoughts
- Visualizations of role turns per conversation

The script analyzes:
- Function call distribution across datasets
- Presence of thought processes before function calls
- Conversation structure and turn distribution
