import importlib.util
import sys
import unittest
from pathlib import Path

from schema.action.api import ApiAction
from schema.observation.text import TextObservation

# Add the parent directory to the path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import the module directly using importlib
spec = importlib.util.spec_from_file_location(
    "raw_to_standardized", str(Path(__file__).parent.parent / "raw_to_standardized.py")
)
raw_to_standardized = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raw_to_standardized)
convert_step = raw_to_standardized.convert_step


class TestFunctionExtraction(unittest.TestCase):
    def test_simple_function_extraction(self):
        """Test extraction of a simple function call."""
        step = {
            "role": "assistant",
            "content": "Let me execute this command:\n\n<function=bash>\n<parameter=command>ls -la</parameter>\n</function>",
        }

        result = convert_step(step)

        # Should have 2 items: a text observation and an API action
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], TextObservation)
        self.assertIsInstance(result[1], ApiAction)

        # Check the API action
        api_action = result[1]
        self.assertEqual(api_action.function, "execute_bash")
        self.assertEqual(api_action.kwargs, {"command": "ls -la"})

        # Check that the full function call was captured
        self.assertIsNotNone(api_action.function_call)
        self.assertIn("<function=bash>", api_action.function_call)
        self.assertIn("</function>", api_action.function_call)

    def test_nested_function_extraction(self):
        """Test extraction of nested function calls."""
        step = {
            "role": "assistant",
            "content": 'Let me execute these commands:\n\n<function=bash>\n<parameter=command>find . -name "*.py" | grep "test"</parameter>\n</function>\n\nAnd now another command:\n\n<function=bash>\n<parameter=command>echo "Testing <function=something>nested</function> functions"</parameter>\n</function>',
        }

        result = convert_step(step)

        # Should have 4 items: text, API action, text, API action
        self.assertEqual(len(result), 4)
        self.assertIsInstance(result[0], TextObservation)
        self.assertIsInstance(result[1], ApiAction)
        self.assertIsInstance(result[2], TextObservation)
        self.assertIsInstance(result[3], ApiAction)

        # Check the first API action
        api_action1 = result[1]
        self.assertEqual(api_action1.function, "execute_bash")
        self.assertEqual(api_action1.kwargs, {"command": 'find . -name "*.py" | grep "test"'})

        # Check that the full function call was captured
        self.assertIsNotNone(api_action1.function_call)
        self.assertIn("<function=bash>", api_action1.function_call)
        self.assertIn("</function>", api_action1.function_call)

        # Check the second API action
        api_action2 = result[3]
        self.assertEqual(api_action2.function, "execute_bash")
        self.assertEqual(
            api_action2.kwargs,
            {"command": 'echo "Testing <function=something>nested</function> functions"'},
        )

        # Check that the full function call was captured
        self.assertIsNotNone(api_action2.function_call)
        self.assertIn("<function=bash>", api_action2.function_call)
        self.assertIn("</function>", api_action2.function_call)

    def test_complex_nested_function_extraction(self):
        """Test extraction of complex nested function calls with the same function name."""
        step = {
            "role": "assistant",
            "content": 'Let me execute this complex command:\n\n<function=bash>\n<parameter=command>echo "This is a <function=bash>nested</function> function call with <function=bash>multiple <function=bash>levels</function> of</function> nesting"</parameter>\n</function>',
        }

        result = convert_step(step)

        # Should have 2 items: a text observation and an API action
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], TextObservation)
        self.assertIsInstance(result[1], ApiAction)

        # Check the API action
        api_action = result[1]
        self.assertEqual(api_action.function, "execute_bash")
        self.assertEqual(
            api_action.kwargs,
            {
                "command": 'echo "This is a <function=bash>nested</function> function call with <function=bash>multiple <function=bash>levels</function> of</function> nesting"'
            },
        )

        # Check that the full function call was captured
        self.assertIsNotNone(api_action.function_call)
        self.assertIn("<function=bash>", api_action.function_call)
        self.assertIn("</function>", api_action.function_call)


if __name__ == "__main__":
    unittest.main()
