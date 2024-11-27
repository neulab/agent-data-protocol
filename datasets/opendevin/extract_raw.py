import os
import sys

input_jsonl = sys.argv[1] if len(sys.argv) > 1 else "feedback_neubig_public.jsonl"
assert os.path.exists(input_jsonl), f"File not found: {input_jsonl}"

with open(input_jsonl, "r") as f:
    for line in f:
        print(line.strip())