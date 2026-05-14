import json
import signal

from datasets import load_dataset

signal.signal(signal.SIGPIPE, signal.SIG_DFL)

dataset = load_dataset("AweAI-Team/Scale-SWE-Distilled", split="train", streaming=True)

for idx, item in enumerate(dataset):
    item["id"] = f"{item['data_source']}_{idx}"
    print(json.dumps(item, ensure_ascii=False))
