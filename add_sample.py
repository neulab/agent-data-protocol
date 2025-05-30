import os
import random
import json
#random.seed(42)

def add_sample(dataset):
    sft = f"datasets/{dataset}/full_sft.jsonl"
    with open(sft) as f: 
        sft = []
        for line in f.readlines():
            try: sft.append(json.loads(line))
            except: 
                print(line)
                continue
    print(f'sft: {len(sft)}')
    out_sft = random.sample(sft, 5)
    #out_sft = sft[:10]
    with open(f"datasets/{dataset}/sample_sft.json", 'w') as f: 
        json.dump(out_sft, f, indent=2, ensure_ascii=False)
    #assert 1==2
    
    print('std')
    std = f"datasets/{dataset}/full_std.jsonl"
    with open(std) as f: std = [json.loads(line) for line in f.readlines()]
    print(f'std: {len(std)}')
    out_std = []
    for sample in out_sft:
        for row in std:
            if str(row['id']) == str(sample['id']):
                out_std.append(row)
    with open(f"datasets/{dataset}/sample_std.json", 'w') as f: 
        json.dump(out_std, f, indent=2, ensure_ascii=False)
    
    print('raw')
    raw = f"datasets/{dataset}/full_raw.jsonl"
    with open(raw) as f: raw = [json.loads(line) for line in f.readlines()]
    print(f'raw: {len(raw)}')
    out_raw = []
    for sample in out_sft:
        for i, row in enumerate(raw):
            if 'id' in row and str(row['id']) == str(sample['id']):
                out_raw.append(row)
            elif str(i) == str(sample['id']):
                out_raw.append(row)
    with open(f"datasets/{dataset}/sample_raw.json", 'w') as f: 
        json.dump(out_raw, f, indent=2, ensure_ascii=False)

def save_sample():
    dirs = os.listdir('datasets')
    print(dirs)
    out = []
    for dir in dirs:
        sample_sft = f"datasets/{dir}/sample_sft.json"
        if not os.path.exists(sample_sft) or dir == 'turkingbench': continue
        print(dir)
        with open(sample_sft) as f: 
            f = json.load(f)
        for line in f: line['source'] = dir
        out += f
    with open('sample_sft.json', 'w') as f: json.dump(out, f, indent=2, ensure_ascii=False)

save_sample()
assert 1==2
dirs = ['openhands']
for d in dirs:
    #if os.path.exists(f"datasets/{d}/full_sft.jsonl"):
    print(d)
    add_sample(d)