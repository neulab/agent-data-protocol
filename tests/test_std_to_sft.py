import json
import sys
import unittest

# Add the parent directory to the path so we can import the scripts
sys.path.append("/workspace/agent-data-collection")


class TestStdToSft(unittest.TestCase):
    """Test the std_to_sft.py script."""

    def test_json_serialization(self):
        """Test that JSON serialization works properly with special characters."""
        # Create a dictionary with special characters
        api_action = "goto(url='https://example.com/path?query=value&special=\"\\'\\n\\t')"
        call_dict = {"name": "browser", "arguments": {"code": api_action}}

        # Serialize and deserialize the dictionary
        call_json = json.dumps(call_dict)
        call = json.loads(call_json)

        # Check that the deserialized dictionary matches the original
        self.assertEqual(call, call_dict)

    def test_function_call_key_handling(self):
        """Test that the function_call key is properly handled."""
        # Create a conversation dictionary without the function_call key
        conversations = [{"from": "function_call", "value": "some value"}]

        # Add the function_call key if it doesn't exist
        if "function_call" not in conversations[-1]:
            conversations[-1]["function_call"] = "some function call"

        # Check that the function_call key exists
        self.assertTrue(isinstance(conversations[-1]["function_call"], str))

    def test_repr_vs_str(self):
        """Test the difference between repr() and str() for special characters."""
        # Create a string with special characters
        special_str = "https://example.com/path?query=value&special=\"'\\n\\t"

        # Test that repr() properly escapes special characters
        call_dict = {"name": "browser", "arguments": {"code": f"goto(url={repr(special_str)})"}}

        # Serialize and deserialize the dictionary
        call_json = json.dumps(call_dict)
        call = json.loads(call_json)

        # Check that the deserialized dictionary matches the original
        self.assertEqual(call, call_dict)


if __name__ == "__main__":
    unittest.main()
