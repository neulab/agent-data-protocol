import os
import re
import unittest


class TestStdToSftFromParameter(unittest.TestCase):
    """Test that all function calls in SFT data use the correct 'from' parameter."""

    def test_finish_function_from_parameter_in_code(self):
        """Test that the 'finish' function uses 'from': 'function_call' in the code."""
        # Read the std_to_sft.py file
        with open(os.path.join(os.path.dirname(__file__), "../scripts/std_to_sft.py"), "r") as f:
            code = f.read()

        # Check that the finish function uses 'from': 'function_call'
        finish_pattern = (
            r'finish_function_call = format_function\([^)]+\)\s+return \{"from": "([^"]+)"'
        )
        match = re.search(finish_pattern, code, re.DOTALL)

        self.assertIsNotNone(match, "Finish function pattern not found in code")
        self.assertEqual(
            match.group(1), "function_call", "Finish function should use 'from': 'function_call'"
        )


if __name__ == "__main__":
    unittest.main()
