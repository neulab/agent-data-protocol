#!/usr/bin/env python3
import json
import sys
from typing import Any, Dict

from schema.action.api import ApiAction
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory


def parse_observation(content: str) -> Dict[str, Any]:
    # Extract content between OBJECTIVE and PREVIOUS ACTIONS
    objective_start = content.find("OBJECTIVE:")
    previous_actions_start = content.find("PREVIOUS ACTIONS:")

    if objective_start != -1 and previous_actions_start != -1:
        # Start after "OBJECTIVE:" and remove any leading/trailing whitespace
        objective_content = content[
            objective_start + len("OBJECTIVE:") : previous_actions_start
        ].strip()
    else:
        objective_content = ""

    axtree = content.split("OBSERVATION:")[1].split("\nURL: ")[0].strip()
    url_content = content.split("\nURL: ")[1].split("\n")[0].strip()

    return {
        "accessibility_tree": axtree,
        "url": url_content,
        "objective": objective_content,
    }


def parse_action(content: str) -> Dict[str, Any]:
    # Extract the action from the assistant's response
    action_start = content.find("```") + 3
    action_end = content.rfind("```")
    action = content[action_start:action_end].strip()
    redundant_thought_part = "In summary, the next action I will perform is"
    thought = content[: action_start - 3].replace(redundant_thought_part, "").strip()

    if not thought:
        thought = None

    # TODO: type is not a browsergym action, but cannot readily be converted as the closest equivalent is fill but does not support pressing enter after typing.
    if "type" in action:
        id = action.split("[")[1].split("]")[0]
        content = action.split("[")[2].split("]")[0]
        if action.count("[") > 3:
            press_enter_after = action.split("[")[3].split("]")[0]
        else:
            press_enter_after = "0"
        fn_name = "type"
        fn_kwargs = {"bid": id, "text": content, "press_enter_after": press_enter_after}
    elif "press" in action:
        key_comb = action.split("[")[1].split("]")[0]
        fn_name = "keyboard_press"
        fn_kwargs = {"key": key_comb}
    elif "scroll" in action:
        direction = action.split("[")[1].split("]")[0]
        if direction == "down":
            fn_name = "scroll"
            fn_kwargs = {"dx": 0, "dy": 500}
        elif direction == "up":
            fn_name = "scroll"
            fn_kwargs = {"dx": 0, "dy": -500}
    elif "close_tab" in action:
        fn_name = "tab_close"
        fn_kwargs = {}
    elif "stop" in action:
        fn_name = "send_msg_to_user"
        fn_kwargs = {"text": action.split("[")[1].split("]")[0]}
    elif "click" in action:
        id = action.split("[")[1].split("]")[0]
        fn_name = "click"
        fn_kwargs = {"bid": id}
    elif "hover" in action:
        id = action.split("[")[1].split("]")[0]
        fn_name = "hover"
        fn_kwargs = {"bid": id}
    elif "new_tab" in action:
        fn_name = "new_tab"
        fn_kwargs = {}
    elif "tab_focus" in action:
        id = action.split("[")[1].split("]")[0]
        fn_name = "tab_focus"
        fn_kwargs = {"index": id}
    elif "goto" in action:
        url = action.split("[")[1].split("]")[0]
        fn_name = "goto"
        fn_kwargs = {"url": url}
    elif "go_back" in action:
        fn_name = "go_back"
        fn_kwargs = {}
    elif "go_forward" in action:
        fn_name = "go_forward"
        fn_kwargs = {}
    else:
        raise ValueError(f"Unknown action: {action}")

    return ApiAction(
        function=fn_name,
        kwargs=fn_kwargs,
        description=thought,
    )


def process_step(step):
    obs_msg = step["messages"][1]
    action_msg = step["messages"][2]

    processed_msgs = []

    obs_data = parse_observation(obs_msg["content"])

    processed_msgs.append(
        WebObservation(
            url=obs_data["url"],
            axtree=obs_data["accessibility_tree"],
            html=None,
            image_observation=None,
            viewport_size=None,
        )
    )

    processed_msgs.append(parse_action(action_msg["content"]))

    return obs_data["objective"], processed_msgs


def main():
    traj_id = -1
    traj_content = []
    traj_goal = None

    for line in sys.stdin:
        step = json.loads(line)

        curr_traj_id = step["id"]
        if traj_id != -1 and traj_id != curr_traj_id:
            goal_message = TextObservation(content=traj_goal, source="user")
            traj_content = [goal_message] + traj_content

            traj = Trajectory(
                id=str(traj_id), content=traj_content, details={"source": "nnetnav-live"}
            )
            print(json.dumps(traj.model_dump()))
            traj_content = []
            traj_goal = None

        try:
            traj_goal, step_msgs = process_step(step)
            traj_content.extend(step_msgs)
            traj_id = curr_traj_id
        except Exception as e:
            print(f"Error processing step {step['id']}: {e}", file=sys.stderr)
            continue

    goal_message = TextObservation(content=traj_goal, source="user")
    traj_content = [goal_message] + traj_content

    traj = Trajectory(id=str(traj_id), content=traj_content, details={"source": "nnetnav-live"})
    print(json.dumps(traj.model_dump()))


if __name__ == "__main__":
    main()
