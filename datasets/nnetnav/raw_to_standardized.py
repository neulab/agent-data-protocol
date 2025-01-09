#!/usr/bin/env python3
import json
import sys
from schema_raw import NNetNavTrajectory
from typing import List, Dict, Any

def parse_observation(content: str) -> Dict[str, Any]:
    # Split content into sections
    sections = content.split('\nOBSERVATION:\n', 1)[-1].split('\nURL:')
    
    # Parse tabs and accessibility tree from first section
    tree_section = sections[0].split('\n')
    tabs = []
    accessibility_tree = []
    in_tree = False
    for line in tree_section:
        if line.startswith('Tab '):
            tabs.append(line.strip())
        elif line.startswith('['):
            in_tree = True
            accessibility_tree.append(line.strip())
        elif in_tree and line.strip():
            accessibility_tree.append(line.strip())
    
    # Remove empty lines from accessibility tree
    accessibility_tree = [line for line in accessibility_tree if line]
    
    # Parse URL, objective and previous actions from second section
    url = None
    objective = None
    previous_actions = []
    
    if len(sections) > 1:
        remaining = sections[1]
        url = remaining.split('\n', 1)[0].strip()
        
        for line in remaining.split('\n'):
            if line.startswith('OBJECTIVE:'):
                objective = line.split(': ', 1)[1]
            elif line.startswith('PREVIOUS ACTIONS:'):
                continue
            elif line.strip() and line[0].isdigit() and ':' in line:
                try:
                    action_num, action = line.split(': ', 1)
                    if action != 'None':
                        previous_actions.append(action)
                except ValueError:
                    continue
    
    return {
        'tabs': tabs,
        'accessibility_tree': accessibility_tree,
        'url': url,
        'objective': objective,
        'previous_actions': previous_actions
    }

def parse_action(content: str) -> str:
    # Extract the action from the assistant's response
    action_start = content.find('```') + 3
    action_end = content.rfind('```')
    return content[action_start:action_end]

def convert_trajectory(traj: NNetNavTrajectory) -> Dict[str, Any]:
    # Initialize standardized format
    standardized = {
        'id': traj.id,
        'source': 'nnetnav',
        'task': None,  # Will be set from first observation
        'steps': []
    }
    
    # Skip system message and process pairs of user/assistant messages
    for i in range(1, len(traj.messages), 2):
        if i + 1 >= len(traj.messages):
            break
            
        obs_msg = traj.messages[i]
        action_msg = traj.messages[i + 1]
        
        if obs_msg.role != 'user' or action_msg.role != 'assistant':
            continue
            
        # Parse observation
        obs_data = parse_observation(obs_msg.content)
        
        # Set task from first observation
        if standardized['task'] is None and obs_data['objective'] is not None:
            standardized['task'] = obs_data['objective']
        
        # Parse action
        action = parse_action(action_msg.content)
        
        # Add step
        standardized['steps'].append({
            'observation': {
                'accessibility_tree': obs_data['accessibility_tree'],
                'url': obs_data['url'],
                'previous_actions': obs_data['previous_actions']
            },
            'action': action
        })
    
    return standardized

def main():
    for line in sys.stdin:
        traj = NNetNavTrajectory.model_validate_json(line)
        standardized = convert_trajectory(traj)
        print(json.dumps(standardized))

if __name__ == "__main__":
    main()