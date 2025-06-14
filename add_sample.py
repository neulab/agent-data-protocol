import json
import os
import random

# random.seed(42)


def add_sample(dataset):
    sft = f"datasets/{dataset}/full_sft.jsonl"
    with open(sft) as f:
        sft = []
        for line in f.readlines():
            try:
                sft.append(json.loads(line))
            except Exception:
                print(line)
                assert 1 == 2
                continue
    print(f"sft: {len(sft)}")
    out_sft = random.sample(sft, min(5, len(sft)))
    # out_sft = sft[:10]
    with open(f"datasets/{dataset}/sample_sft.json", "w") as f:
        json.dump(out_sft, f, indent=2, ensure_ascii=False)
    # assert 1==2

    print("std")
    std = f"datasets/{dataset}/full_std.jsonl"
    with open(std) as f:
        std = [json.loads(line) for line in f.readlines()]
    print(f"std: {len(std)}")
    out_std = []
    for sample in out_sft:
        for row in std:
            if str(row["id"]) == str(sample["id"]):
                out_std.append(row)
    with open(f"datasets/{dataset}/sample_std.json", "w") as f:
        json.dump(out_std, f, indent=2, ensure_ascii=False)

    print("raw")
    raw = f"datasets/{dataset}/full_raw.jsonl"
    with open(raw) as f:
        raw = [json.loads(line) for line in f.readlines()]
    print(f"raw: {len(raw)}")
    out_raw = []
    for sample in out_sft:
        for i, row in enumerate(raw):
            if "id" in row and str(row["id"]) == str(sample["id"]):
                out_raw.append(row)
            elif str(i) == str(sample["id"]):
                out_raw.append(row)
            elif "instance_id" in row and str(row["instance_id"]) == str(sample["id"]):
                out_raw.append(row)
    with open(f"datasets/{dataset}/sample_raw.json", "w") as f:
        json.dump(out_raw, f, indent=2, ensure_ascii=False)


def save_sample():
    dirs = os.listdir("datasets")
    print(dirs)
    out = []
    for dir in dirs:
        sample_sft = f"datasets/{dir}/sample_sft.json"
        if not os.path.exists(sample_sft) or dir == "turkingbench":
            continue
        print(dir)
        with open(sample_sft) as f:
            f_content = json.load(f)
        for line in f_content:
            line["source"] = dir
        out += f_content
    with open("sample_sft.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


# Comment out the assertion to allow the script to run
# save_sample()
# assert 1 == 2
dirs = [os.getenv("MY_DATASET", "SWE-Gym_OpenHands-Sampled-Trajectories")]
for d in dirs:
    # if os.path.exists(f"datasets/{d}/full_sft.jsonl"):
    print(d)
    add_sample(d)
