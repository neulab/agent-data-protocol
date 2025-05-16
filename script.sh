#!/bin/bash

export MY_DATASET=synatra #SWE-Gym_OpenHands-Sampled-Trajectories
echo $MY_DATASET

# Step 1
python datasets/$MY_DATASET/extract_raw.py > datasets/$MY_DATASET/full_raw.jsonl

# Step 2
export PYTHONPATH=`pwd`:$PYTHONPATH
cat datasets/$MY_DATASET/full_raw.jsonl | python datasets/$MY_DATASET/raw_to_standardized.py > datasets/$MY_DATASET/full_std.jsonl

# Step 3
#cat datasets/$MY_DATASET/full_std.jsonl | python -u scripts/std_to_sft.py --is_web=no --chunk=all > datasets/$MY_DATASET/full_sft.jsonl
