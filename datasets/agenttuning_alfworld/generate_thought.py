import json
import os
import re
import sys

import openai

api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    print("openai api_key not found", file=sys.stderr)
client = openai.OpenAI(api_key=api_key)

EXAMPLES = """
    EXAMPLE 1:
    History:
    [{
        "class_": "text_observation",
        "content": "Here is your task. You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a garbagecan 1, a handtowelholder 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.\nYour task is to: find two spraybottle and put them in toilet.",
        "source": "user"
    }]

    Current Action: api_action
    Function: go
    Arguments: {"location": "\"shelf 1\""}

    Model Output:
    {"description": "To accomplish the task, I need to first find the spraybottles. I'll start with checking the shelves, as they are common places to store spraybottles."}

    EXAMPLE 2:
    History:
    [{
        "class_": "text_observation",
        "content": "On the toilet 1, you see nothing.",
        "source": "user"
    }]

    Current Action: api_action
    Function: put
    Arguments: {"item": "\"spraybottle 1\"", "target": "\"toilet 1\""}

    Model Output:
    {"description": "The previous action seems invalid as the script reads \"On the toilet 1, you see nothing.\" I should try putting the spraybottle again in the toilet."}
"""


def generate_thought(context, action_class, action_function, action_kwargs):
    prompt = f"""
    You are helping an agent operating in a simulated household environment.
    Based on the history and current action, generate a reasoning of why the agent decides to perform this action.
    Below are some example:
    {EXAMPLES}

    Now, consider the following:

    History:
    {context}

    Current Action: {action_class}
    Function: {action_function}
    Arguments: {action_kwargs}

    Respond **only** in valid JSON format with a single field "description".
    """

    response = client.chat.completions.create(
        model="o4-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    match = re.search(r'\{.*?"description"\s*:\s*".*?"\s*\}', content, re.DOTALL)
    if match:
        try:
            description_obj = json.loads(match.group(0))
            return description_obj["description"]
        except Exception as e:
            print("JSON parsing failed:", e, file=sys.stderr)
            print("Matched content:", match.group(0), file=sys.stderr)
            return ""
    else:
        print("No valid JSON found in GPT response:", content, file=sys.stderr)
        return ""
