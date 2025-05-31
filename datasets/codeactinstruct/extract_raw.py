import json

from datasets import load_dataset

ds = load_dataset("xingyaoww/code-act")

# CodeAct dataset
codeact_ds = ds["codeact"]
for sample in codeact_ds:
    print(json.dumps(sample))

# # General dataset
# general_ds = ds["general"]
# for sample in general_ds:
#     print(json.dumps(sample))
