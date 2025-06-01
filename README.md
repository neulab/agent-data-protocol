# agent-data-collection

This is a repository for agent training data collection by CMU, OSU, and HKU.

- `datasets/`: Contains datasets, each with at least the following elements
  - `README.md`: A description of the dataset
  - `sample_raw.json`: 2-5 raw samples from the corpus in the original format
  - `extract_raw.py`: A script that extracts a raw jsonl file from the corpus
  - If `raw_to_standardized.py` exists (a script that converts the raw jsonl file to our standardized format jsonl):
    - `sample_std.json`: 2-5 samples from the corpus in our standardized format
  - If `sample_std.json` exists:
    - `sample_sft.json`: 2-5 samples from the corpus in the SFT format
  - Note: There should not be other .json files checked in to the dataset directory
- `schema/`: Contains schema definitions for the standardized format
- `scripts/`:
  - `jsonl_to_indented_json.py`: Converts a jsonl file to an indented json file for easier viewing
  - `json_to_jsonl.py`: Converts a json file to a jsonl file
  - `std_to_sft.py`: Converts standardized format to SFT format
  - `install_hooks.sh`: Script to install pre-commit hooks

## Development Setup

### Pre-commit Hooks

This repository uses pre-commit hooks to ensure code quality. To install the hooks, run:

```bash
./scripts/install_hooks.sh
```

This will install pre-commit and set up the hooks defined in `.pre-commit-config.yaml`.

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

Once we have our standardized format, we will create a script that converts, line-by-line, a jsonl file in the raw format to one in the standardized format in `raw_to_standardized.py`.

We can then apply this to the sample data to create a sample in the standardized format.

```bash
export MY_DATASET=dataset_name
export PYTHONPATH=`pwd`:$PYTHONPATH
cat datasets/$MY_DATASET/sample_raw.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/raw_to_standardized.py | python scripts/jsonl_to_indented_json.py > datasets/$MY_DATASET/sample_std.json
```

### Step 3: Convert Standardized Format to SFT Format

Once we have the standardized format, we can convert it to the SFT format:

```bash
export MY_DATASET=dataset_name
export PYTHONPATH=`pwd`:$PYTHONPATH
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python -u scripts/std_to_sft.py --is_web=no --chunk=all --keep_system=yes > datasets/$MY_DATASET/sample_sft.json
```

Use `--is_web=yes` for web-based datasets like `mind2web, synatra`.

Run the validator script on the dataset to ensure that it is in the correct format:

```bash
pytest tests/test_raw_schemas.py tests/test_standardized_schemas.py
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
export MY_DATASET=dataset_name
export PYTHONPATH=`pwd`:$PYTHONPATH
cat datasets/$MY_DATASET/full_std.jsonl | python -u scripts/std_to_sft.py --is_web=no --chunk=all --keep_system=yes > datasets/$MY_DATASET/full_sft.jsonl
```

Use `--is_web=yes` for web-based datasets like `mind2web, synatra`.
