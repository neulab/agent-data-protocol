import json

from datasets import concatenate_datasets, get_dataset_config_names, load_dataset

# Load the Toucan-1.5M dataset from Hugging Face
# ds = load_dataset("Agent-Ark/Toucan-1.5M", "SFT", split="train")
repo = "Agent-Ark/Toucan-1.5M"
configs = get_dataset_config_names(repo)

ds = concatenate_datasets([load_dataset(repo, cfg, split="train") for cfg in configs])

# Process the samples
for id, sample in enumerate(ds):
    # Create a unique ID
    sample_id = f"toucan_sample_{id}"

    # Extract the relevant fields from the Toucan dataset
    raw_sample = {
        "id": sample_id,
        "uuid": sample.get("uuid", ""),
        "subset_name": sample.get("subset_name", ""),
        "messages": sample.get("messages", []),
        "question": sample.get("question", ""),
        "available_tools": sample.get("available_tools", "[]")
        if "available_tools" in sample
        else sample.get("tools", "[]"),
        "target_tools": sample.get("target_tools", []),
        "question_quality_assessment": sample.get("question_quality_assessment", ""),
        "response_quality_assessment": sample.get("response_quality_assessment", ""),
        "metadata": sample.get("metadata", {}),
    }

    print(json.dumps(raw_sample))
