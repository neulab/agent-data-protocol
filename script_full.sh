export MY_DATASET=swe-smith
mkdir -p datasets/$MY_DATASET/full_sft

echo RAW
python datasets/$MY_DATASET/extract_raw.py > datasets/$MY_DATASET/full_raw.jsonl
export PYTHONPATH=`pwd`:$PYTHONPATH

echo STD
cat datasets/$MY_DATASET/full_raw.jsonl | python datasets/$MY_DATASET/raw_to_standardized.py > datasets/$MY_DATASET/full_std.jsonl

echo SFT
cat datasets/$MY_DATASET/full_std.jsonl | python agents/openhands/std_to_sft.py --is_web=no --api_env=execute_bash > datasets/$MY_DATASET/full_sft/full_sft_openhands.jsonl
cat datasets/$MY_DATASET/full_std.jsonl | python agents/sweagent/std_to_sft.py > datasets/$MY_DATASET/full_sft/full_sft_openhands.jsonl
# cat datasets/$MY_DATASET/full_std.jsonl | python agents/agentlab/std_to_sft.py > datasets/$MY_DATASET/full_sft/full_sft_agentlab.jsonl
