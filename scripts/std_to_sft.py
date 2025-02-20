import json
import os
import sys
import argparse
import traceback

from browsergym.core.action.highlevel import HighLevelActionSet

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.image import ImageObservation
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation

from schema.trajectory import Trajectory

import function_calling as codeact_function_calling

from scripts.html_to_axtree import HTMLToAXTree
from scripts.browsing_prompts.system import get_system_message
from scripts.browsing_prompts.user import get_web_user_message

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

tools = codeact_function_calling.get_tools(
            codeact_enable_browsing= True,
            codeact_enable_jupyter=True,
            codeact_enable_llm_editor=True,
        )

dataset = os.getenv("MY_DATASET")
assert dataset, "Please set the environment variable MY_DATASET"

action_function = {
    'python': 'execute_ipython_cell',
    'bash': 'execute_bash',
    'web': 'broswer'
}

function_args = {
    'execute_ipython_cell': 'code',
    'execute_bash': 'command',
}

USE_NAV = (
    os.environ.get('USE_NAV', 'true') == 'true'
)  # only disable NAV actions when running webarena and miniwob benchmarks

generate_axtree = HTMLToAXTree(dataset)

parser = argparse.ArgumentParser(description='Convert standardized data to SFT format')
# parser.add_argument('--output_dataset', type=str, help='Output Dataset name', default='sample_sft.json')
parser.add_argument('--chunk', type=str, help='Dataset name', required=True)
parser.add_argument('--is_web', type=str, choices=['yes', 'no'], help='Is Dataset type web api', required=True)
args = parser.parse_args()

def standardized_event_to_openhands_message(id, event: ApiAction | CodeAction | MessageAction | TextObservation | ImageObservation | WebObservation, details: dict, previous_actions: list) -> dict:
    # NOTE for KETAN: deal with the different types of events later
    # The Web and API Actions are based on Browsergym's schema. So use normal actions if the style is different to HTML/AXTree
    if isinstance(event, WebObservation):
        if event.axtree is not None:
            axtree = event.axtree
        elif generate_axtree.last_html != event.html:
            axtree = generate_axtree.build_axtree(id, event.html, args.chunk)
        else:
            axtree = generate_axtree.last_xtree
        prompt = get_web_user_message("", event.url, axtree, previous_actions)
        return {"from": "observation", "value": prompt}
    
    if isinstance(event, ApiAction):
        thought = "THOUGHT: " + event.description + "\n\n" if event.description else ""

        if event.function == 'goto': # could add more or condtions here for actions that don't require bid
            api_action = f"{event.function}({', '.join([f'{k}={v}' for k, v in event.kwargs.items() if k not in ['element_id', 'xpath']])})"
            previous_actions.extend([api_action])
            return {"from": "function_call", "value": f"{thought}", "function_call": f"{{\"name\": \"browser\", \"arguments\": {{\"code\": \"{api_action}\"}}}}"}

        arguments = None
        # try to directly get the browsergym_id from the event kwargs
        browsergym_id = event.kwargs.get('element_id', None)

        # this gets the browsergym_id of the element that the user is interacting with
        # the latest(last seen) html's obs is updated whenever build_axtree is called
        # the latest obs is used to get the browsergym_id
        if not browsergym_id:
            event_xpath = event.kwargs.get('xpath', None)
            if event_xpath:
                browsergym_id = generate_axtree.get_bid(id, event_xpath, args.chunk)
        # for tool calls that are not browser based
        if not browsergym_id:
            arguments = {k: v for k, v in event.kwargs.items() if k not in ['element_id', 'xpath']}
            api_action = f"{event.function}({', '.join([f'{k}={v}' for k, v in arguments.items()])})"
        # for tool calls that are browser based
        elif len(event.kwargs)==1 and 'element_id' in event.kwargs:
            api_action = f"{event.function}(bid={browsergym_id})"
        else:
            api_action = f"{event.function}(bid={browsergym_id}, {', '.join([f'{k}={v}' for k, v in event.kwargs.items() if k not in ['element_id', 'xpath']])})"
        previous_actions.extend([api_action])
        for tool in tools:
            if event.function == tool['name']:
                return {"from": "function_call", "value": f"{thought}", "function_call": f"{{\"name\": \"{event.function}\", \"arguments\": {json.dumps(arguments)}}}"}
        return {"from": "function_call", "value": f"{thought}", "function_call": f"{{\"name\": \"browser\", \"arguments\": {{\"code\": \"{api_action}\"}}}}"}

    if isinstance(event, CodeAction):

        thought = "THOUGHT: " + event.description + "\n\n" if event.description else ""
        function_name = action_function.get(event.language, f'execute_{event.language}')
        arg = function_args.get(function_name, 'code')
        return {"from": "function_call", "value": f"{thought}", "function_call": f"{{\"name\": \"{function_name}\", \"arguments\": {{\"{arg}\": \"{event.content}\"}}}}"}
    
    elif isinstance(event, MessageAction):
        thought = event.description + "\n\n" if event.description else ""
        return {"from": "gpt", "value": f"{thought}{event.content}"}
    
    elif isinstance(event, TextObservation):
        # I had this earlier to include source in the message, but OpenHands does not have that and has bash executions as user messages
        #return {"role": event.source, "content": event.content} if event.source == "user" or event.source=='system' else {"role": "user", "content": f"OBSERVATION from {event.source}: {event.content}"}

        if event.source == 'user':
            event.source = 'human'
        
        if event.source == 'assistant':
            event.source = 'gpt'

        return {"from": event.source, "value": event.content} if event.source in ["human", "gpt"] else {"from": "observation", "value":  f"{event.content}"}

    else:
        raise ValueError(f"Unknown event type: {type(event)}\n{event}")

def process_row(line):
    try:
    # if True:
        sft_data = []
        std_dataset = [json.loads(line)]
        for std_data in std_dataset:
            trajectory = Trajectory(**std_data)
            id = trajectory.id
            events = trajectory.content
            details = trajectory.details

            conversations = []
            previous_actions = []

            # Add system message similar to OH Browsing Agent if the dataset is web dataset
            if args.is_web=='yes':
                action_subsets = ['chat', 'bid']
                if USE_NAV:
                    action_subsets.append('nav')
                action_space = HighLevelActionSet(
                    subsets=action_subsets,
                    strict=False,  # less strict on the parsing of the actions
                    multiaction=True,  # enable to agent to take multiple actions at once
                )
            for event in events:
                if hasattr(event, 'source') and event.source == 'system': # Ignore dataset specific system messages since we have a unified system prompt
                    continue
                try: 
                    message = standardized_event_to_openhands_message(id, event, details, previous_actions)
                    if len(conversations) == 0: 
                        conversations.extend([message])
                        continue
                    # code to process multiple consecutive function calls + observations
                    elif message['from'] == 'function_call' and conversations[-1]['from'] == 'function_call':
                        conversations[-1]['value'] = conversations[-1]['value'] + '\n' + message['value'].replace('THOUGHT: ', '')
                        # if the previous event contains only one function call
                        if isinstance(conversations[-1]['function_call'], str): 
                            conversations[-1]['function_call'] = [conversations[-1]['function_call'], message['function_call']]
                        # if the previous event already contains multiple function calls
                        elif isinstance(conversations[-1]['function_call'], list): conversations[-1]['function_call'].append(message['function_call'])
                        else: raise ValueError(f"Unknown function_call type: {type(conversations[-1]['function_call'])}\n{conversations[-1]['function_call']}")
                        continue
                    if conversations[-1]['from'] == 'function_call' and isinstance(event, TextObservation):
                        message['from'] = 'observation'
                    conversations.extend([message])
                except Exception as e: 
                    traceback.print_exc()
                    print(e)
                    return None

            system_message = get_system_message()
            return {"id": trajectory.id, "conversations": conversations, 'system': system_message,'tools': json.dumps(tools)}
    except Exception as e: 
        traceback.print_exc()
        print(e)
        return None

with open(f'datasets/{dataset}/full_std.jsonl') as f: data = f.readlines()

output = []
# for i in range(len(data)):
#     line = data[i]
#     output_line = process_row(line)
#     if output_line is not None: 
#         output.append(output_line)
#         print(i)
with ThreadPoolExecutor(max_workers=64) as executor:   
    futures = [
        executor.submit(process_row, line)
        for line in data
    ]
    for future in tqdm(as_completed(futures), total=len(futures)):
        output_line = future.result()
        if output_line is not None: output.append(output_line)

# remove duplicate values
temp = []
for i in range(len(output)):
    appeared = False
    for j in range(0, i):
        if output[i] == output[j]: 
            appeared = True
            break
    if not appeared: temp.append(output[i])
output = temp

print(len(output))
with open(f'datasets/{dataset}/{dataset}.json', 'w') as f: json.dump(output, f, indent=2, ensure_ascii=False)
