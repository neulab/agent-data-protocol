export MY_DATASET=go-browse-wa
mkdir -p datasets/$MY_DATASET/sample_sft

echo RAW
# python datasets/$MY_DATASET/extract_raw.py | python scripts/jsonl_to_json.py  > datasets/$MY_DATASET/sample_raw.json
export PYTHONPATH=`pwd`:$PYTHONPATH

echo STD
cat datasets/$MY_DATASET/sample_raw.json | python scripts/json_to_jsonl.py | python datasets/$MY_DATASET/raw_to_standardized.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_std.json

echo SFT
# openhands
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/openhands/std_to_sft.py --is_web=yes --api_env=browser | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft/sample_sft_openhands.json
# sweagent
# cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/sweagent/std_to_sft.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft/sample_sft_sweagent.json
# agentlab
cat datasets/$MY_DATASET/sample_std.json | python scripts/json_to_jsonl.py | python agents/agentlab/std_to_sft.py | python scripts/jsonl_to_json.py > datasets/$MY_DATASET/sample_sft/sample_sft_agentlab.json
