# Description: Contains the system messages for the browsing prompts.
# Source: https://github.com/All-Hands-AI/OpenHands/blob/8c32ef2234eed4963f3d6dec1dcb4f6c8baa5e9e/agenthub/browsing_agent/browsing_agent.py#L43C1-L55C4

def get_web_system_message(goal: str, action_space: str) -> str:
    return f"""\
# Instructions
Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.

# Goal:
{goal}

# Action Space
{action_space}
"""
