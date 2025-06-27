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
- `std_to_sft_scripts_swe-agent/`: Contains specialized scripts for SWE agent datasets
  - `std_to_sft_SWE_smith.py`: Converts standardized format to SFT format specifically for SWE agent datasets (SWE-Gym_OpenHands-Sampled-Trajectories, SWE-smith_5kTrajectories, code_feedback, codeactinstruct, nebius_SWE-agent-trajectories, openhands)

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

Once you have created the sample raw data, you need to write a converter to transform it into our standardized format. The standardized format is defined in the `schema/` directory.

To create a converter, create a file called `raw_to_standardized.py` in your dataset directory. This script should:

1. Read the raw data from stdin
2. Convert it to the standardized format
3. Output the standardized data to stdout

The standardized format should follow these guidelines:
- TextObservation.source must be one of: 'user', 'agent', or 'environment'
- ImageObservation.source must be one of: 'user', 'agent', or 'environment'
- WebObservation must include a url field
- WebObservation.viewport_size should be a list, not a tuple
- WebObservation should include axtree (can be None)

You can run the following command to create a sample standardized file:

```bash
python datasets/$MY_DATASET/raw_to_standardized.py < datasets/$MY_DATASET/sample_raw.json > datasets/$MY_DATASET/sample_std.json
```

#### (Optional) Generate Function Thoughts from Standardized Data
This step enriches the standardized dataset by adding a thoughts for each action that doesn't have thoughts, using in-context learning examples and a language model to simulate the agent's reasoning.

```bash
export MY_DATASET=dataset_name
export OPENAI_API_KEY=your_key_here
cat datasets/$MY_DATASET/full_std.jsonl | python scripts/generate_thoughts_std.py
```

This script will:

Read all entries from datasets/$MY_DATASET/full_std.jsonl and all saved thoughts from datasets/$MY_DATASET/generated_thoughts.json, generate thoughts if thoughts are missing and not generated already in generated_thoughts.json, save the generated thoughts to datasets/$MY_DATASET/generated_thoughts.json, and lastly write an updated full_std.jsonl that includes the descriptions in-place

Note: If thoughts already exist in generated_thoughts.json, the script will skip regeneration for those entries. Delete the file if you wish to regenerate from scratch.


#### Validate the Standardized Format

To ensure your standardized format is correct, run the validation test:

```bash
python -m pytest tests/test_standardized_schemas.py::test_sample_standardized_against_schema[/workspace/agent-data-collection/datasets/$MY_DATASET/sample_std.json] -v
```

If the test passes, your standardized format is correct. If not, you'll need to fix the issues in your `raw_to_standardized.py` script.

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
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python -u scripts/std_to_sft.py --is_web=no --chunk=all --api_env=execute_ipython_cell > datasets/$MY_DATASET/sample_sft.json
```

Use `--is_web=yes` for web-based datasets like `mind2web, synatra`.
Use `--chunk=all` by default.
Use `--api_env=browser` for web-based datasets like `mind2web, synatra`.

#### Alternative: Using the SWE Agent Specialized Script

For SWE agent datasets (SWE-Gym_OpenHands-Sampled-Trajectories, SWE-smith_5kTrajectories, code_feedback, codeactinstruct, nebius_SWE-agent-trajectories, openhands), you can use the specialized conversion script:

```bash
export MY_DATASET=dataset_name
cat datasets/$MY_DATASET/sample_std.json | python std_to_sft_scripts_swe-agent/std_to_sft_SWE_smith.py > datasets/$MY_DATASET/sample_sft.json
```

This script is specifically designed for SWE agent datasets and handles their unique format requirements.

Run the validator script on the dataset to ensure that it is in the correct format:

```bash
pytest tests/test_raw_schemas.py tests/test_standardized_schemas.py
```

### Step 4: Write README

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

#### Alternative: Using the SWE Agent Specialized Script for Full Data

For SWE agent datasets, you can also use the specialized conversion script for full data:

```bash
export MY_DATASET=dataset_name
cat datasets/$MY_DATASET/full_std.jsonl | python std_to_sft_scripts_swe-agent/std_to_sft_SWE_smith.py > datasets/$MY_DATASET/full_sft.jsonl
```
