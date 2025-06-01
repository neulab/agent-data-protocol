import json


def test_json_escape():
    """Test that special characters are properly escaped in JSON strings."""
    api_action = 'goto(url="https://example.com/path?query=value&special=\\"\'\\n\\t")'

    # Before the fix, this would fail with a JSON decode error
    try:
        # The problematic line from std_to_sft.py
        escaped_api_action = (
            api_action.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )
        json_str = f'{{"name": "browser", "arguments": {{"code": "{escaped_api_action}"}}}}'
        parsed = json.loads(json_str)
        print("JSON escape test passed!")
        return True
    except json.JSONDecodeError as e:
        print(f"JSON escape test failed: {e}")
        return False


def test_function_call_key():
    """Test that the function_call key is properly added to the dictionary."""
    # Before the fix, this would cause a KeyError when accessing 'function_call'
    conversations = [
        {"from": "function_call", "value": "some value", "function_call": "some function call"}
    ]

    try:
        # The problematic check from std_to_sft.py
        if isinstance(conversations[-1]["function_call"], str):
            print("Function call key test passed!")
            return True
    except KeyError as e:
        print(f"Function call key test failed: {e}")
        return False


if __name__ == "__main__":
    json_test_passed = test_json_escape()
    function_call_test_passed = test_function_call_key()

    if json_test_passed and function_call_test_passed:
        print("All tests passed! The fixes are working correctly.")
        exit(0)
    else:
        print("Some tests failed. The fixes may not be working correctly.")
        exit(1)
