import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))

input_jsonl = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, "feedback-public.jsonl")
assert os.path.exists(input_jsonl), f"File not found: {input_jsonl}"

with open(input_jsonl, "r") as f:
    for line in f:
        print(line.strip())