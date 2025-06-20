import json
import sys

from schema_raw import SchemaRaw
from trajectory import Trajectory as synatra_trajectory

from schema.action.api import ApiAction
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory


def convert_step(step: synatra_trajectory) -> tuple[WebObservation, ApiAction]:
    web_observation = WebObservation(
        html=None,
        axtree=step.obs + "\n\n# Previous Actions\n" + step.history.replace("element_id", "bid"),
        image_observation=None,
        viewport_size=None,
        url=step.website,
    )

    kwargs = {}
    function = step.next_action.action_type.to_string().lower()
    if function == "stop":
        kwargs["answer"] = f'"{step.next_action.typed_string}"'
    elif function == "type":
        kwargs["bid"] = f'"{step.next_action.axt_node_id}"'
        kwargs["text"] = f'"{step.next_action.typed_string}"'
    elif function in ["click"]:
        kwargs["bid"] = f'"{step.next_action.axt_node_id}"'
    elif function == "scroll":
        kwargs["delta_x"] = 0
        kwargs["delta_y"] = 100 if "down" in step.next_action.typed_string else -100
    elif function in ["key_press", "press"]:
        kwargs["key_comb"] = f'"{step.next_action.typed_string}"'
        function = "press"
    elif function in ["new_tab", "goto", "goto_url"]:
        kwargs["url"] = f'"{step.next_action.typed_string}"'
        function = "goto" if function == "goto_url" else function
    elif function in ["tab_focus", "page_focus", "switch_tab"]:
        kwargs["page_number"] = step.next_action.typed_string
        function = "tab_focus"
    elif function in ["go_back", "page_close", "go_forward", "close_tab"]:
        kwargs = {}
        function = "tab_close" if function == "page_close" else function
    else:
        raise ValueError(f"Unknown function: {function}")

    # description = {
    #     "sub_task": step.next_action.subtask,
    #     "CoT": step.next_action.cot,
    #     "action_description": step.next_action.action_description,
    # }
    description = ""
    if step.next_action.subtask:
        description += f"{step.next_action.subtask}.\n\n"
    if step.next_action.cot:
        description += f"{step.next_action.cot}\n\n"
    if step.next_action.action_description:
        description += f"{step.next_action.action_description}\n\n"
    api_action = ApiAction(function=function, kwargs=kwargs, description=description)
    return web_observation, api_action


if __name__ == "__main__":
    # Process each line of input individually
    idx = 0
    for line in sys.stdin:
        raw_data = json.loads(line)
        raw_data = SchemaRaw(**raw_data)

        try:
            data = synatra_trajectory(raw_data)
        except Exception as e:
            print(f"Error processing trajectory {idx}: {e}", file=sys.stderr)
            idx += 1
            continue
        objective = f"Generate the next action given the an objective, the current web page, and previous actions. \n\nObjective: {data.objective}"
        content: list = [TextObservation(content=objective, source="user")]

        content.extend(convert_step(data))

        standardized_data = Trajectory(
            id=str(idx),
            content=content,
            details={
                "task_description": data.objective,
                "website": data.website,
            },
        )

        # Print the standardized data as JSON
        print(json.dumps(standardized_data.model_dump()))
        idx += 1
