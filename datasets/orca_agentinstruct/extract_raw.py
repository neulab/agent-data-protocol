import json

from datasets import load_dataset

ds = load_dataset("microsoft/orca-agentinstruct-1M-v1")


for category in ds:
    cat_data = ds[category]
    # if category != "code_":
    #     continue
    for id, sample in enumerate(cat_data):
        sample["id"] = f"orca_{category}{id}"
        sample["conversations"] = json.loads(sample["messages"])
        del sample["messages"]
        print(json.dumps(sample))
