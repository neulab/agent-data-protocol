#!/usr/bin/env python3
import json
import sys
from typing import Any, Dict

from schema_raw import SchemaRaw


def parse_observation(content: str) -> Dict[str, Any]:
    # Split content into sections
    sections = content.split("\nOBSERVATION:\n", 1)[-1].split("\nURL:")

    # Parse tabs and accessibility tree from first section
    tree_section = sections[0].split("\n")
    tabs = []
    accessibility_tree = []
    in_tree = False
    for line in tree_section:
        if line.startswith("Tab "):
            tabs.append(line.strip())
        elif line.startswith("["):
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
        url = remaining.split("\n", 1)[0].strip()

        for line in remaining.split("\n"):
            if line.startswith("OBJECTIVE:"):
                objective = line.split(": ", 1)[1]
            elif line.startswith("PREVIOUS ACTIONS:"):
                continue
            elif line.strip() and line[0].isdigit() and ":" in line:
                try:
                    action_num, action = line.split(": ", 1)
                    if action != "None":
                        previous_actions.append(action)
                except ValueError:
                    continue

    return {
        "tabs": tabs,
        "accessibility_tree": accessibility_tree,
        "url": url,
        "objective": objective,
        "previous_actions": previous_actions,
    }


def parse_action(content: str) -> Dict[str, Any]:
    # Extract the action from the assistant's response
    action_start = content.find("```") + 3
    action_end = content.rfind("```")
    action_str = content[action_start:action_end].strip()
    redundant_thought_part = "In summary, the next action I will perform is"
    thought_str = content[: action_start - 3].replace(redundant_thought_part, "").strip()

    if not thought_str:
        thought_str = None

    noop_action = {"type": "api_action", "function": "noop", "kwargs": {}, "description": None}

    # Handle empty or malformed actions
    if not action_str:
        return noop_action

    # Parse action string into components
    parts = action_str.split(" ", 1)
    function = parts[0]
    kwargs = {}

    # Handle invalid function names
    if function.startswith("[") or function not in [
        "type",
        "click",
        "hover",
        "press",
        "scroll",
        "new_tab",
        "tab_focus",
        "close_tab",
        "goto",
        "go_back",
        "go_forward",
        "stop",
        "noop",
    ]:
        return noop_action

    try:
        if len(parts) > 1:
            # Handle special cases where the action doesn't have brackets
            if function in ["new_tab", "close_tab", "go_back", "go_forward", "noop"]:
                pass
            else:
                # Split by '] [' but handle the case where there's only one argument
                args_str = parts[1].strip("[]")
                if "] [" in args_str:
                    args = args_str.split("] [")
                else:
                    args = [args_str]

                # Clean up args - remove any text after the ID number
                args = [arg.split()[0] if arg and arg[0].isdigit() else arg for arg in args]

                if function == "type":
                    kwargs = {
                        "id": args[0],
                        "text": args[1] if len(args) > 1 else "",
                        "press_enter_after": args[2] == "1" if len(args) > 2 else True,
                    }
                elif function == "click":
                    kwargs = {"id": args[0]}
                elif function == "hover":
                    kwargs = {"id": args[0]}
                elif function == "press":
                    kwargs = {"key_comb": args[0]}
                elif function == "scroll":
                    kwargs = {"direction": args[0]}
                elif function == "tab_focus":
                    try:
                        kwargs = {"tab_index": int(args[0])}
                    except ValueError:
                        return {"type": "api_action", "function": "noop", "kwargs": {}}
                elif function == "stop":
                    kwargs = {"answer": args[0] if args else None}
                elif function == "goto":
                    kwargs = {"url": args[0]}
    except (IndexError, ValueError):
        # If there's any error parsing the action, return a noop
        return noop_action

    return {
        "type": "api_action",
        "function": function,
        "kwargs": kwargs,
        "description": thought_str,
    }


def convert_trajectory(traj: SchemaRaw) -> Dict[str, Any]:
    # Initialize standardized format
    standardized = {
        "id": traj.id,
        "content": [],
        "details": {
            "source": "nnetnav",
            "task": None,  # Will be set from first observation
        },
    }

    # Skip system message and process pairs of user/assistant messages
    for i in range(1, len(traj.messages), 2):
        if i + 1 >= len(traj.messages):
            break

        obs_msg = traj.messages[i]
        action_msg = traj.messages[i + 1]

        if obs_msg.role != "user" or action_msg.role != "assistant":
            continue

        # Parse observation
        obs_data = parse_observation(obs_msg.content)

        # Set task from first observation
        if standardized["details"]["task"] is None and obs_data["objective"] is not None:
            standardized["details"]["task"] = obs_data["objective"]

        # Add observation
        standardized["content"].append(
            {
                "type": "web_observation",
                "html": None,
                "url": obs_data["url"],
                "axtree": "\n".join(obs_data["accessibility_tree"]),
                "image_observation": None,
                "viewport_size": None,
            }
        )

        # Parse and add action
        action = parse_action(action_msg.content)
        standardized["content"].append(action)

    return standardized


def main():
    # Process each line as a separate JSON object
    for line in sys.stdin:
        item = json.loads(line)
        traj = SchemaRaw.model_validate(item)
        standardized = convert_trajectory(traj)
        # Print each result as a separate line
        print(json.dumps(standardized))


if __name__ == "__main__":
    main()
