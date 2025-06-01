import json

from datasets import load_dataset


def deserialize_to_list_of_dicts(serialized_data):
    num_items = len(serialized_data[next(iter(serialized_data))])

    list_of_dicts = []
    for i in range(num_items):
        item_dict = {key: value[i] for key, value in serialized_data.items()}
        list_of_dicts.append(item_dict)

    return list_of_dicts


data = load_dataset("screenagent_dataset.py", split="train", cache_dir="./")["data"]

data_for_demonstrate = [deserialize_to_list_of_dicts(data_item) for data_item in data[:5]]

for data_item in data_for_demonstrate:
    for step_item in data_item:
        step_item["screenshot"] = "<image>"

with open("sample_raw.json", "w") as f:
    json.dump(data_for_demonstrate, f, indent=4)
