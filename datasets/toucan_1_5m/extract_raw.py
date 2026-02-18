import json

from datasets import load_dataset

# Load the Toucan-1.5M dataset from Hugging Face
# For sample generation, we'll limit to just a few samples from the Kimi-K2 config
ds = load_dataset("Agent-Ark/Toucan-1.5M", "Kimi-K2", split="train")

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
        "available_tools": sample.get("available_tools", []),
        "target_tools": sample.get("target_tools", []),
        "question_quality_assessment": sample.get("question_quality_assessment", ""),
        "response_quality_assessment": sample.get("response_quality_assessment", ""),
        "metadata": sample.get("metadata", {}),
    }
   
    if not raw_sample["question_quality_assessment"]:
        continue
    try: 
        response_quality_assessment = json.loads(raw_sample["response_quality_assessment"])
    except: 
        continue
    if "desired_tools_used_percentage" in response_quality_assessment:
        if response_quality_assessment["desired_tools_used_percentage"] != 1:
            continue
        if not response_quality_assessment["order_correctness"]:
            continue

    print(json.dumps(raw_sample))
