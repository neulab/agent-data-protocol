# Description: This file contains the prompt messages for the user.
# Source: https://github.com/All-Hands-AI/OpenHands/blob/8c32ef2234eed4963f3d6dec1dcb4f6c8baa5e9e/agenthub/browsing_agent/browsing_agent.py#L68C1-L92C1

import os

USE_CONCISE_ANSWER = (
    os.environ.get('USE_CONCISE_ANSWER', 'false') == 'true'
)  # only return concise answer when running webarena and miniwob benchmarks

CONCISE_INSTRUCTION = """\

Here is another example with chain of thought of a valid action when providing a concise answer to user:
"
In order to accomplish my goal I need to send the information asked back to the user. This page list the information of HP Inkjet Fax Machine, which is the product identified in the objective. Its price is $279.49. I will send a message back to user with the answer.
```send_msg_to_user("$279.49")```
"
"""


def get_web_user_message(
    error_prefix: str, cur_url: str, cur_axtree_txt: str, prev_action_str: str
) -> str:
    previous_actions = f"# Previous Actions\n{prev_action_str}" if not "\n# Previous Actions" in cur_axtree_txt else ""
    prompt = f"""\
{error_prefix}

# Current Page URL:
{cur_url}

# Current Accessibility Tree:
{cur_axtree_txt}

{previous_actions}

Here is an example with chain of thought of a valid action when clicking on a button:
"
THOUGHT: In order to accomplish my goal I need to click on the button with bid 12

ACTION:
```click("12")```
"

If you do not have a proper THOUGHT for your action, you can directly provide the ACTION.
""".strip()
    if USE_CONCISE_ANSWER:
        prompt += CONCISE_INSTRUCTION
    return prompt
