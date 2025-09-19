#!/bin/bash
#SBATCH --job-name=std_to_owl
#SBATCH --partition=cpu
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source .venv/bin/activate

datasets=(
    # "code_feedback:0.1" # DONE
    # "nebius_SWE-agent-trajectories:0.2" # DONE
    "orca_agentinstruct:0.001" # MISSING
    # "agenttuning_alfworld:1.0" # DONE
    # "agenttuning_db:1.0" # DONE
    # "agenttuning_kg:1.0" # ERRORS - Multiple tasks per trajectory (OWL does a single one)
    # "agenttuning_mind2web:1.0" # DONE
    # "agenttuning_os:1.0" # ERRORS - Multiple tasks per trajectory (OWL does a single one)
    # "agenttuning_webshop:1.0" # DONE
    # "codeactinstruct:1.0" # ERRORS - Multiple user feedback/submission steps (No user feedback in OWL)
    # "swe-gym_openhands_sampled_trajectories:1.0" # ERRORS - Strange endings involving multiple message actions that don't do much.
    "swe-smith:1.0" # MISSING

    # "openhands" - web, don't use
    # "nnetnav-live" - web, don't use
    # "synatra" - web, don't use
    # "nnetnav-wa" - web, don't use
    # "go-browse-wa" - web, don't use
    # "mind2web" - web, don't use
)

# Loop through each dataset
for dataset_entry in "${datasets[@]}"; do
    # Split dataset name and ratio
    IFS=':' read -r dataset ratio <<< "$dataset_entry"

    echo "Processing dataset: $dataset (ratio: $ratio)"

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
        --llm_model "litellm_proxy/gpt-4o-mini" \
        --use_templates \
        --random_seed 42 \
        --sample_ratio "$ratio"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully processed $dataset"
    else
        echo "Error processing $dataset"
    fi
    
    echo "---"
done

echo "All datasets processed!"
