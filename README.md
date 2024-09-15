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
