import json
import os
import sys
import argparse

from browsergym.core.action.highlevel import HighLevelActionSet

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.image import ImageObservation
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation

from schema.trajectory import Trajectory

from scripts.html_to_axtree import HTMLToAXTree
from scripts.browsing_prompts.system import get_web_system_message
from scripts.browsing_prompts.user import get_web_user_message

dataset = os.getenv("MY_DATASET")

USE_NAV = (
    os.environ.get('USE_NAV', 'true') == 'true'
)  # only disable NAV actions when running webarena and miniwob benchmarks

generate_axtree = HTMLToAXTree(dataset)

parser = argparse.ArgumentParser(description='Convert standardized data to SFT format')
# parser.add_argument('--input_dataset', type=str, help='Input Dataset name', default='sample.json')
# parser.add_argument('--output_dataset', type=str, help='Output Dataset name', default='sample_sft.json')
parser.add_argument('--chunk', type=str, help='Dataset name', required=True)
parser.add_argument('--is_web', type=str, choices=['yes', 'no'], help='Is Dataset type web api', required=True)
args = parser.parse_args()

def standardized_event_to_openhands_message(id, event: ApiAction | CodeAction | MessageAction | TextObservation | ImageObservation | WebObservation, details: dict, previos_actions: list) -> dict:
    # NOTE for KETAN: deal with the different types of events later
    if isinstance(event, WebObservation):
        if event.axtree is not None:
            axtree = event.axtree
        elif generate_axtree.last_html != event.html:
            axtree = generate_axtree.build_axtree(id, event.html, args.chunk)
        else:
            axtree = generate_axtree.last_xtree
        prompt = get_web_user_message("", event.url, axtree, previos_actions)
        return {"role": "user", "content": [{'type': 'text', 'text': prompt}]}
    
    if isinstance(event, ApiAction):
        if event.function == 'goto': # could add more or condtions here for actions that don't require bid
            api_action = f"{event.function}({', '.join([f'{k}={v}' for k, v in event.kwargs.items() if k not in ['element_id', 'xpath']])})"
            previos_actions.extend([api_action])
            return {"role": "assistant", "content": f"ACTION: \n```{api_action}```\n"}

        # try to directly get the browsergym_id from the event kwargs
        browsergym_id = event.kwargs.get('element_id', None)

        # this gets the browsergym_id of the element that the user is interacting with
        # the latest(last seen) html's obs is updated whenever build_axtree is called
        # the latest obs is used to get the browsergym_id
        if not browsergym_id:
            event_xpath = event.kwargs.get('xpath', None)
            if event_xpath:
                browsergym_id = generate_axtree.get_bid(id, event_xpath, args.chunk)
        if len(event.kwargs)==1:
            api_action = f"{event.function}(bid={browsergym_id})"
        else:
            api_action = f"{event.function}(bid={browsergym_id}, {', '.join([f'{k}={v}' for k, v in event.kwargs.items() if k not in ['element_id', 'xpath']])})"
        previos_actions.extend([api_action])
        thought = "THOUGHT: " + event.description + "\n\n" if event.description else ""
        return {"role": "assistant", "content": f"{thought}ACTION: \n```{api_action}```\n"}

    if isinstance(event, CodeAction):
        if 'python' in event.language:
            event.language = 'ipython'
        return {"role": "assistant", "content": f"{event.description}\n<execute_{event.language}>\n{event.content}\n</execute_{event.language}>"}
    
    elif isinstance(event, MessageAction):
        # print(event.content)
        return {"role": "assistant", "content": f"THINK: {event.description}\nACT: {event.content}"} if event.description else {"role": "assistant", "content": f"ACT: {event.content}"}
    
    elif isinstance(event, TextObservation):
        # I had this earlier to include source in the message, but OpenHands does not have that and has bash executions as user messages
        #return {"role": event.source, "content": event.content} if event.source == "user" or event.source=='system' else {"role": "user", "content": f"OBSERVATION from {event.source}: {event.content}"}
        return {"role": event.source, "content": event.content} if event.source == "user" or event.source=='system' else {"role": "user", "content": [{'type': 'text', 'text': f"OBSERVATION:\n{event.content}"}]}

    else:
        raise ValueError(f"Unknown event type: {type(event)}\n{event}")
    
sft_data = []

for line in sys.stdin:
    std_dataset = [json.loads(line)]
    for std_data in std_dataset:
        # print(trajectory_data)
        trajectory = Trajectory(**std_data)
        # Process the trajectory object 
        # print(trajectory)
        # iterate over the event in the trajectory
        
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
            system_message = get_web_system_message(action_space.describe(with_long_description=False,
                                                                           with_examples=True))
            conversations.extend([{"role": "system", "content": system_message}])

        for event in events:
            conversations.extend([standardized_event_to_openhands_message(id, event, details, previous_actions)])

        print(json.dumps({"id": trajectory.id, "conversations": conversations}))


# with open(f'./datasets/{dataset}/{args.output_dataset}', 'w') as file:
#     json.dump(sft_data, file, indent=2)

# print(sft_data)