import argparse
import json
import sys
import os

from schema.action.api import ApiAction
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory
from schema_raw import SchemaRaw
from trajectory import Trajectory as synatra_trajectory

def convert_step(step: synatra_trajectory) -> tuple[WebObservation, ApiAction]:
    web_observation = WebObservation(
        html=step.obs,
        image_observation=None,
        viewport_size=None,
        url=None,
    )

    kwargs = {}
    function=step.next_action.action_type.to_string().lower()
    if function == "stop":
        kwargs["answer"] = step.next_action.typed_string
    elif function == "type":
        kwargs["text"] = step.next_action.typed_string
        kwargs["element_id"] = step.next_action.axt_node_id
    elif function in ["click"]:
        kwargs["element_id"] = step.next_action.axt_node_id
    elif function == "scroll":
        kwargs["dx"] = 0
        kwargs["dy"] = 100 if "down" in step.next_action.typed_string else -100
    elif function in ["key_press", "press"]:
        kwargs["key_comb"] = step.next_action.typed_string
        function = "press"
    elif function in ["new_tab", "goto", "goto_url"]:
        kwargs["url"] = step.next_action.typed_string
        function = "goto" if function == "goto_url" else function
    elif function in ['tab_focus', 'page_focus', 'switch_tab']:
        kwargs["page_number"] = step.next_action.typed_string
        function = "tab_focus"
    elif function in ['go_back', 'page_close', 'go_forward','close_tab']:
        kwargs = {}
        function = "tab_close" if function == "page_close" else function
    else:
        raise ValueError(f"Unknown function: {function}")
    
    description = {
        "sub_task": step.next_action.subtask,
        "CoT": step.next_action.cot,
        "action_description": step.next_action.action_description,
        "history": step.history,
    }
    api_action = ApiAction(
        function=function,
        kwargs=kwargs,
        description=json.dumps(description)
    )
    return web_observation, api_action


if __name__ == "__main__":

    idx = 0

    for line in sys.stdin:

        raw_data = json.loads(line)
        raw_data = SchemaRaw(**raw_data)

        try:
            data = synatra_trajectory(raw_data)
        except:
            continue
        
        content: list = [
            TextObservation(
                content=data.objective, source="user"
            )
        ]
        content.extend(convert_step(data))

        standardized_data = Trajectory(
            id=str(idx),
            content=content,
        )
        
        idx+=1
        
        print(standardized_data.model_dump_json())
