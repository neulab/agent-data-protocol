import os
import sys
import json
from glob import glob

exp_name = 'release-32b-it1-trainlite-temp_1.0-fp8_7'

script_dir = os.path.dirname(os.path.realpath(__file__))
if not os.path.exists(os.path.join(script_dir, exp_name)):
    print('Running ./extract_raw.sh', file=sys.stderr)
    os.system('./extract_raw.sh')

successful_trajectories = os.path.join(script_dir, exp_name, 'dataset.openai.jsonl')
assert os.path.exists(successful_trajectories), f'Failed to extract {successful_trajectories}'

prompt_logs = os.path.join(script_dir, exp_name, 'prompt_logs')
assert os.path.exists(prompt_logs), f'Failed to extract {prompt_logs}'

successful_instance_ids = set()
with open(successful_trajectories, 'r') as f:
    for line in f:
        sub_trajectory_file = json.loads(line)
        successful_instance_ids.add(sub_trajectory_file['instance_id'])

for instance_dir in glob(os.path.join(prompt_logs, '*/')):
    instance_id = os.path.basename(instance_dir.rstrip('/'))
    trajectory = {
        'messages': [],
        'instance_id': instance_id,
        'exp_name': exp_name,
        'fail': instance_id not in successful_instance_ids,
    }
    for sub_trajectory_file in sorted(glob(os.path.join(instance_dir, '*.json'))):
        with open(sub_trajectory_file, 'r') as f:
            sub_trajectory = json.load(f)
            if not sub_trajectory['completion']:
                sub_trajectory['completion'] = []
            trajectory['messages'] += sub_trajectory['messages'] + sub_trajectory['completion']
    print(json.dumps(trajectory))