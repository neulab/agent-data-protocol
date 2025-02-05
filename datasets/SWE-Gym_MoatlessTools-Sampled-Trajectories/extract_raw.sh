#!/bin/bash -x

cd $(dirname $0)

[ -f release-32b-it1-trainlite-temp_1.0-fp8-30runs.zip ] || wget https://huggingface.co/datasets/SWE-Gym/MoatlessTools-Sampled-Trajectories/resolve/main/32b/release-32b-it1-trainlite-temp_1.0-fp8-30runs.zip

unzip release-32b-it1-trainlite-temp_1.0-fp8-30runs.zip 'home/jiayipan/code/24FA/release-SWE-Gym/release/32b/release-32b-it1-trainlite-temp_1.0-fp8-30runs/release-32b-it1-trainlite-temp_1.0-fp8_7/dataset.openai.jsonl'

unzip release-32b-it1-trainlite-temp_1.0-fp8-30runs.zip 'home/jiayipan/code/24FA/release-SWE-Gym/release/32b/release-32b-it1-trainlite-temp_1.0-fp8-30runs/release-32b-it1-trainlite-temp_1.0-fp8_7/prompt_logs/*'

mv home/jiayipan/code/24FA/release-SWE-Gym/release/32b/release-32b-it1-trainlite-temp_1.0-fp8-30runs/release-32b-it1-trainlite-temp_1.0-fp8_7/ .