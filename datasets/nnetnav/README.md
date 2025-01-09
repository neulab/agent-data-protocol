# NNetNav

[NNetNav](https://github.com/MurtyShikhar/NNetNav) is a dataset of 6K web navigation demonstrations collected using an interaction-first method for generating demonstrations for web-agents on any website. The dataset contains demonstrations of web navigation tasks across multiple websites.

# Steps to Process

1. **Install Dependencies**
     ```sh
     pip install -r requirements.txt
     ```

2. **Extract raw data**
     ```sh
     python extract_raw.py > nnetnav_raw.jsonl
     ```

3. **Convert raw data to standardized format**
     ```sh
     cat nnetnav_raw.jsonl | python raw_to_standardized.py > nnetnav_standardized.jsonl
     ```