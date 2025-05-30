import json
import random

out = []
with open('/project/flame/yueqis/agent-data-collection/datasets/agenttuning/full_sft.jsonl') as f: f = f.readlines()
temp = random.choices(f,k=100)
temp = [json.loads(line) for line in temp]
for line in temp:
    line['source'] = 'agenttuning'
out += temp
with open('/project/flame/yueqis/agent-data-collection/datasets/code_feedback/full_sft.jsonl') as f: f = f.readlines()
temp = random.choices(f,k=100)
temp = [json.loads(line) for line in temp]
for line in temp:
    line['source'] = 'code_feedback'
out += temp
with open('/project/flame/yueqis/agent-data-collection/datasets/codeactinstruct/full_sft.jsonl') as f: f = f.readlines()
temp = random.choices(f,k=100)
temp = [json.loads(line) for line in temp]
for line in temp:
    line['source'] = 'codeactinstruct'
out += temp
with open('/project/flame/yueqis/agent-data-collection/datasets/nnetnav/full_sft.jsonl') as f: f = f.readlines()
temp = random.choices(f,k=100)
temp = [json.loads(line) for line in temp]
for line in temp:
    line['source'] = 'nnetnav'
out += temp
with open('/project/flame/yueqis/agent-data-collection/datasets/orca_agentinstruct/full_sft.jsonl') as f: f = f.readlines()
temp = random.choices(f,k=100)
temp = [json.loads(line) for line in temp]
for line in temp:
    line['source'] = 'orca_agentinstruct'
out += temp
with open('/project/flame/yueqis/agent-data-collection/datasets/synatra/full_sft.jsonl') as f: f = f.readlines()
temp = random.choices(f,k=100)
temp = [json.loads(line) for line in temp]
for line in temp:
    line['source'] = 'synatra'
out += temp
with open('/project/flame/yueqis/agent-data-collection/datasets/openhands/full_sft.jsonl') as f: f = f.readlines()[:-1]
temp = []
random.shuffle(f)
for line in f:
    if len(temp) >= 100: break
    try:
        temp.append(json.loads(line))
    except: continue
for line in temp:
    line['source'] = 'openhands'
out += temp
print(len(temp))
out = random.choices(out,k=50)
with open('test.json', 'w') as f: json.dump(out, f, indent=2)