import os
import re


def test_finish_function_from_parameter_in_code():
    """Test that the 'finish' function uses 'from': 'function_call' in the code."""
    # Read the std_to_sft.py file
    with open(os.path.join(os.path.dirname(__file__), "../scripts/std_to_sft.py"), "r") as f:
        code = f.read()

    # Check that the finish function exists
    finish_function_pattern = r"finish_function_call = format_function\("
    function_match = re.search(finish_function_pattern, code)
    assert function_match is not None, "Finish function call not found in code"

    # Check that there's a return with "from": "function_call" after the finish function
    finish_return_pattern = r'return\s+\{\s*"from":\s*"([^"]+)"'
    return_match = re.search(finish_return_pattern, code[function_match.end() :], re.DOTALL)

    assert return_match is not None, "Return statement after finish function not found"
    assert return_match.group(1) == "function_call", (
        "Finish function should use 'from': 'function_call'"
    )
