#!/bin/bash

source /project/flame/yueqis/miniconda3/etc/profile.d/conda.sh
conda activate unified

export MY_DATASET=synatra  # SWE-Gym_OpenHands-Sampled-Trajectories
echo $MY_DATASET

echo Step_1
python datasets/$MY_DATASET/extract_raw.py > datasets/$MY_DATASET/full_raw.jsonl

echo Step_2
export PYTHONPATH=`pwd`:$PYTHONPATH
cat datasets/$MY_DATASET/full_raw.jsonl | python datasets/$MY_DATASET/raw_to_standardized.py > datasets/$MY_DATASET/full_std.jsonl
# Optionally
# cat /project/flame/yueqis/agent-data-collection/datasets/$MY_DATASET/full_std.jsonl | python scripts/generate_thoughts_std.py

echo Step_3
cat datasets/$MY_DATASET/full_std.jsonl | python -u scripts/std_to_sft.py --is_web=yes --chunk=all --api_env=browser

echo Sampling
python add_sample.py
