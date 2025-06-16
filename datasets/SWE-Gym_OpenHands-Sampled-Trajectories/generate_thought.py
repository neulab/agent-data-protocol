import json
import re
import os
import sys

import openai
api_key = os.getenv("OPENAI_API_KEY", "") 
if not api_key:
    print(f"openai api_key not found", file=sys.stderr)
client = openai.OpenAI(api_key=api_key)

STR_REPLACE_EDITOR_EXAMPLES = """
    EXAMPLE 1:
    History:
    [{
        "class_": "text_observation",
        "content": "Here's the result of running `cat -n` on /workspace/dask__dask__2022.6/dask/base.py:\n  1101\t    @nor ......[Truncated]",
        "source": "user"
    }]

    Current Action: api_action
    Function: str_replace_editor
    Arguments: {"command": "insert", "path": "/workspace/dask__dask__2022.6/dask/base.py", "new_str": "from enum import Enum\n\n@normalize_token.register(Enum)\ndef normalize_enum(e):\n    return type(e).__name__, e.name, e.value\n", "insert_line": 965}

    Model Output:
    {"description": "The `dask.base` module contains the `normalize_token` function, which is used to create deterministic tokens for various Python objects. To address the issue described in the PR, we need to add a new normalization function for `Enum` types. This function should be registered with the `normalize_token` dispatcher.\n\nHere's the plan:\n\n1. **Identify the Location for the New Function**: The new function should be added to the `dask.base` module, where other normalization functions are defined.\n\n2. **Implement the Function**: Implement the function `normalize_enum` as described in the PR, which returns a tuple of the enum's type name, name, and value.\n\n3. **Register the Function**: Use the `@normalize_token.register(Enum)` decorator to register the function with the `normalize_token` dispatcher.\n\nLet's implement these changes in the `dask.base` module."}
    
    EXAMPLE 2:
    History:
    [{
        "class_": "text_observation",
        "content": "Here's the files and directories up to 2 levels deep in /workspace/dask__dask__2022.6/dask, excludin ......[Truncated]",
        "source": "user"
    }]

    Current Action: api_action
    Function: str_replace_editor
    Arguments: {"command": "view", "path": "/workspace/dask__dask__2022.6/dask/base.py"}

    Model Output:
    {"description": "The repository appears to be organized into several sub-modules, with the core logic likely within files like `base.py`. The task at hand requires us to modify the Dask library to ensure deterministic hashing for `Enum` types. The `tokenize` function that is mentioned in the PR description is likely defined or used in `base.py`.\n\nLet's follow these steps:\n\n1. **Locate the `base.py` file** - This file likely contains the `tokenize` function, which we'll need to modify to support deterministic hashing for `Enum` types.\n\n2. **Create a script to reproduce the error** - We will write a small script that uses the `tokenize` function on `Enum` types to confirm its current behavior prior to modification. This will help us ensure our changes are effective.\n\n3. **Modify the code in `base.py`** - Implement the logic provided in the PR description to handle `Enum` types in a deterministic manner.\n\n4. **Rerun the test script** - Confirm that the changes resolve the issue of non-deterministic hashing for `Enum` types.\n\nLet's start by viewing the `base.py` file."}
"""

BASH_EXAMPLES = """
    EXAMPLE 1:
    History:
    [{
        "class_": "text_observation",
        "content": "python3 /workspace/python__mypy__0.820/reproduce_error.py[Python Interpreter: /opt/miniconda3/envs/t ......[Truncated]",
        "source": "user"
    }]

    Current Action: code_action
    Function: bash
    Arguments: {"command":"mypy --follow-imports=silent --show-error-codes --warn-unused-ignores --warn-redundant-casts --strict --show-traceback /workspace/python__mypy__0.820/reproduce_error.py"}

    Model Output:
    {"description": "The script executed without any errors, which indicates that the issue might not manifest when running this code as a standalone script. The problem likely occurs when `mypy` performs type checking on this code. \n\nTo reproduce the error, we need to run `mypy` on this script using the specified command-line flags from the PR description. Let's proceed with that."}
    
    EXAMPLE 2:
    History:
    [{
        "class_": "text_observation",
        "content": "The file /workspace/python__mypy__0.820/mypy/constraints.py has been edited. Here's the result of ru ......[Truncated]",
        "source": "user"
    }]

    Current Action: code_action
    Function: bash
    Arguments: {"command":"mypy --follow-imports=silent --show-error-codes --warn-unused-ignores --warn-redundant-casts --strict --show-traceback /workspace/python__mypy__0.820/reproduce_error.py"}

    Model Output:
    {"description": "I have modified the `infer_constraints_from_protocol_members` function in `constraints.py` to handle cases where `inst` or `temp` might be `None`. Instead of asserting their presence, the function now skips over such cases, preventing the internal error.\n\nLet's rerun the `mypy` check on the `reproduce_error.py` script to verify that this change resolves the issue."}
"""

def generate_thought(context, action_class, action_function, action_kwargs):
    if action_class == "api_action" and action_function == "str_replace_editor":
        examples = STR_REPLACE_EDITOR_EXAMPLES
    elif action_class == "code_action" and action_function == "bash":
        examples = BASH_EXAMPLES
    else: raise ValueError(f"{action_class}\n{action_function}")
    prompt = f"""
    You are helping an agent working on code modifications in a simulated development environment.
    Based on the history and current action, generate a reasoning of why the agent decides to perform this action.
    Below are some example: 
    {examples}
    
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
