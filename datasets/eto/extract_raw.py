import json

from huggingface_hub import hf_hub_download

data_files = {
    "webshop": "data/webshop_sft.json",
    "sciword": "data/sciworld_sft.json",
    "alfword": "data/alfworld_sft.json",
}


def load_data(subset):
    dataset = hf_hub_download(
        "agent-eto/eto-sft-trajectory", repo_type="dataset", filename=data_files[subset]
    )
    with open(dataset, "r") as f:
        data = json.load(f)
    for item in data:
        item["id"] = f"{subset}_{item['id']}"
    return data


if __name__ == "__main__":
    data = []
    for subset in data_files:
        data.extend(load_data(subset))

    for item in data:
        print(json.dumps(item))
