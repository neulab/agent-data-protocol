#!/bin/bash

# Array of dataset directories (extracted from the file paths)
datasets=(
    "agenttuning_os"
    "agenttuning_webshop"
    "swe-gym_openhands_sampled_trajectories"
    "agenttuning_db"
    "agenttuning_alfworld"
    "swe-smith"
    "openhands"
    "agenttuning_kg"
    "codeactinstruct"
    # "agenttuning_mind2web" - web
    # "nnetnav-live" - web
    # "synatra" - web
    # "nnetnav-wa" - web
    # "go-browse-wa" - web
    # "mind2web" - web
    # "code_feedback" - too long
    # "orca_agentinstruct" - too long
    # "nebius_SWE-agent-trajectories" - too long
)

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    
    input_file="datasets/$dataset/full_std.jsonl"
    output_dir="datasets/$dataset/full_owl"
    
    # Check if input file exists
    if [ ! -f "$input_file" ]; then
        echo "Warning: Input file $input_file not found, skipping..."
        continue
    fi
    
    # Run the Python script
    python scripts/std_to_owl.py \
        --input_file "$input_file" \
        --output_dir "$output_dir" \
        --llm_model "litellm_proxy/gpt-4o" \
        --use_templates
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully processed $dataset"
    else
        echo "Error processing $dataset"
    fi
    
    echo "---"
done

echo "All datasets processed!"