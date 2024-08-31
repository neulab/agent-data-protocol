# mind2web

[Mind2Web](https://osu-nlp-group.github.io/Mind2Web/) is a dataset for developing and evaluating generalist agents for the web that can follow language instructions to complete complex tasks on any website.

## Downloading the Dataset

Our extraction script is based on the raw dumps from mind2web. In order to download them, you need to use some specialized "globus" software, please follow the [directions on the site](https://github.com/OSU-NLP-Group/Mind2Web?tab=readme-ov-file#raw-dump-with-full-traces-and-snapshots).

## Extracting Data

Once you have downloaded the raw dumps, first install requirements.

```bash
pip install -r datasets/mind2web/requirements.txt
playwright install --with-deps chromium
```

You can extract the data using the following command.

```bash
python datasets/mind2web/extract_raw.py | head -n 3 | python scripts/jsonl_to_indented_json.py > datasets/mind2web/sample_raw.json
```

Then you can use the following command to convert the extracted data to the format used by the dataset, where `PATH_TO_DUMP` is replaced with the actual path to the dump.

```bash
export PATH_TO_DUMP=/path/to/mind2web/dump
cat datasets/mind2web/sample_raw.json | python scripts/json_to_jsonl.py | python datasets/mind2web/raw_to_standardized.py --raw-dump $PATH_TO_DUMP | python scripts/jsonl_to_indented_json.py > datasets/mind2web/sample.json
```
