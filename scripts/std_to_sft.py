import argparse

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.image import ImageObservation
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation

from schema.trajectory import Trajectory
import json

import os

dataset = os.getenv("MY_DATASET")

parser = argparse.ArgumentParser(description='Convert standardized data to SFT format')
parser.add_argument('--input_dataset', type=str, help='Dataset name', default='sample.json')
parser.add_argument('--output_dataset', type=str, help='Dataset name', default='sample_sft.json')
args = parser.parse_args()

def standardized_event_to_openhands_message(event: ApiAction | CodeAction | MessageAction | TextObservation | ImageObservation | WebObservation) -> dict:
    # NOTE for KETAN: deal with the different types of events later
    # if isinstance(event, ApiAction):
    #     return {"role": "assistant", "content": f"{event.description}\n<execute_api>{event.function}("+", ".join([f'{k}="{v}"' for k, v in event.kwargs.items()])+")"}

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

with open(f'./datasets/{dataset}/{args.input_dataset}', 'r') as file:
    trajectory_data_file = json.load(file)
    for trajectory_data in trajectory_data_file:
        # print(trajectory_data)
        trajectory = Trajectory(**trajectory_data)
        # Process the trajectory object 
        # print(trajectory)
        # iterate over the event in the trajectory
        
        id = trajectory.id
        conversations = []
        events = trajectory.content
        for event in events:
            conversations.append(standardized_event_to_openhands_message(event))

        sft_data.append({"id": trajectory.id, "conversations": conversations})


with open(f'./datasets/{dataset}/{args.output_dataset}', 'w') as file:
    json.dump(sft_data, file, indent=2)

# print(sft_data)