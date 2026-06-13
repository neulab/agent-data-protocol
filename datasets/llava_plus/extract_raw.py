import json
import os
import sys
from typing import Any

from huggingface_hub import hf_hub_download


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


# Using load_dataset() directly will lead to issues due to misaligned formats in llava plus
dataset_llava_plus_fname = hf_hub_download(
    repo_id="LLaVA-VL/llava-plus-data",
    repo_type="dataset",
    filename="llava-plus-v1-117k-tool-merge.json",
    local_dir="./",
)

with open(dataset_llava_plus_fname) as f:
    dataset_llava_plus = json.load(f)

try:
    for sample in dataset_llava_plus:
        print(json.dumps(json_safe(sample), ensure_ascii=False))
except BrokenPipeError:
    sys.stdout = open(os.devnull, "w")
    sys.exit(0)
