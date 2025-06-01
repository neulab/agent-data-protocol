import os
import re


def test_finish_function_from_parameter_in_code():
    """Test that the 'finish' function uses 'from': 'function_call' in the code."""
    # Read the std_to_sft.py file
    with open(os.path.join(os.path.dirname(__file__), "../scripts/std_to_sft.py"), "r") as f:
        code = f.read()

    # Check that the finish function uses 'from': 'function_call'
    finish_pattern = r'finish_function_call = format_function\([^)]+\)\s+return \{"from": "([^"]+)"'
    match = re.search(finish_pattern, code, re.DOTALL)

    assert match is not None, "Finish function pattern not found in code"
    assert match.group(1) == "function_call", "Finish function should use 'from': 'function_call'"
