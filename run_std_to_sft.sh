#!/bin/bash

# Create a directory for logs
mkdir -p logs

# Function to determine if a dataset is web-based
is_web_dataset() {
  local dataset=$1
  if [[ "$dataset" == "mind2web" || "$dataset" == "synatra" || "$dataset" == "webarena_successful" || "$dataset" == "weblinx" ]]; then
    echo "yes"
  else
    echo "no"
  fi
}

# Process each dataset
for dataset_path in $(find /workspace/agent-data-collection/datasets -mindepth 1 -maxdepth 1 -type d | sort); do
  dataset=$(basename "$dataset_path")
  echo "Processing $dataset..."
  
  # Check if sample.json exists
  if [ -f "$dataset_path/sample.json" ]; then
    # Determine if this is a web-based dataset
    IS_WEB=$(is_web_dataset "$dataset")
    
    # Run the conversion in the background
    (
      export MY_DATASET=$dataset
      export PYTHONPATH=/workspace/agent-data-collection:$PYTHONPATH
      
      echo "Converting $dataset (web=$IS_WEB)..."
      cat "$dataset_path/sample.json" | python scripts/json_to_jsonl.py | python scripts/std_to_sft.py --is_web=$IS_WEB --chunk=all --keep_system=yes > "$dataset_path/sample_sft.json" 2> "logs/${dataset}_conversion.log"
      
      echo "Finished converting $dataset"
    ) &
    
    # Limit the number of parallel processes to avoid overloading the system
    # Wait if we have too many processes running
    while [ $(jobs -p | wc -l) -ge 4 ]; do
      sleep 1
    done
  else
    echo "Skipping $dataset - no sample.json found"
  fi
done

# Wait for all background processes to finish
wait

echo "All conversions completed!"