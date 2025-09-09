# Description: This file contains the prompt messages for the user.
# Source: https://github.com/All-Hands-AI/OpenHands/blob/8c32ef2234eed4963f3d6dec1dcb4f6c8baa5e9e/agenthub/browsing_agent/browsing_agent.py#L68C1-L92C1


def get_web_user_message(
    last_browser_action_error: str, cur_url: str, cur_axtree_txt: str, focused_element_bid: str
) -> str:
    text = f"[Current URL: {cur_url}]\n"
    text += f"[Focused element bid: {focused_element_bid}]\n\n"
    if last_browser_action_error:
        text += (
            "================ BEGIN error message ===============\n"
            "The following error occurred when executing the last action:\n"
            f"{last_browser_action_error}\n"
            "================ END error message ===============\n"
        )
    else:
        text += "[Action executed successfully.]\n"
    text += (
        f"============== BEGIN accessibility tree ==============\n"
        f"{cur_axtree_txt}\n"
        f"============== END accessibility tree ==============\n"
    )
    return text
