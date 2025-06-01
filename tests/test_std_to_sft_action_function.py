import os
import re


def test_action_function_mapping():
    """Test that the action_function dictionary has correct mappings."""
    # Read the std_to_sft.py file directly
    with open(os.path.join("scripts", "std_to_sft.py"), "r") as f:
        content = f.read()

    # Use regex to find the action_function dictionary definition
    match = re.search(r"action_function\s*=\s*{([^}]+)}", content)
    assert match, "Could not find action_function dictionary in std_to_sft.py"

    # Extract the dictionary content
    dict_content = match.group(1)

    # Check that the web action is correctly spelled as "browser"
    assert '"web": "browser"' in dict_content, (
        "Web action should be mapped to 'browser', not 'broswer'"
    )
