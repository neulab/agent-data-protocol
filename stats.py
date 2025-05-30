import json
dataset = 'synatra'
with open(f'/project/flame/yueqis/agent-data-collection/datasets/{dataset}/full_std.jsonl') as f: 
    f = f.readlines()
print(f"dataset: {dataset}")
out = []
for line in f[:1000]:
    try: out.append(json.loads(line))
    except: continue
total_len_conv = 0
total_num_char = 0
test = 0
for row in out:
    total_len_conv += len(row['content'])
    for m in row['content']:
        try:
            total_num_char += len(f"{m}")
            test += 1
        except: continue
print(f"avg # of round: {total_len_conv/len(out)}")
print(f"avg # of chars: {total_num_char/len(out)}")
print(test)