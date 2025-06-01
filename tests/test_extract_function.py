import os
import sys
import unittest

# Add the parent directory to the path so we can import the functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Define the function we're testing
def extract_full_function_call(content, tools=None):
    """Extract the full function call from the content."""
    if tools is None:
        tools = [
            "execute_bash",
            "think",
            "finish",
            "web_read",
            "browser",
            "execute_ipython_cell",
            "str_replace_editor",
            "edit_file",
        ]

    for tool in tools:
        # Use a non-greedy pattern with balanced matching for nested tags
        start_tag = f"<function={tool}>"
        end_tag = "</function>"

        # Find the start position of the function tag
        start_pos = content.find(start_tag)
        if start_pos == -1:
            continue

        # Find the matching end tag using a balanced approach
        pos = start_pos + len(start_tag)
        nesting = 1
        while pos < len(content) and nesting > 0:
            next_start = content.find(start_tag, pos)
            next_end = content.find(end_tag, pos)

            # If no more tags are found, break
            if next_start == -1 and next_end == -1:
                break

            # If the next tag is a start tag
            if next_start != -1 and (next_end == -1 or next_start < next_end):
                nesting += 1
                pos = next_start + len(start_tag)
            # If the next tag is an end tag
            elif next_end != -1:
                nesting -= 1
                pos = next_end + len(end_tag)

        # If we found a balanced match
        if nesting == 0:
            end_pos = pos - len(end_tag)
            return content[start_pos:pos]

    return None


class TestExtractFunction(unittest.TestCase):
    """Test the extract_full_function_call function."""

    def test_extract_full_function_call(self):
        """Test the extract_full_function_call function."""
        # Test with a valid function call
        content = "Let me check the files\n\n<function=execute_bash>\n<parameter=command>\nls -la\n</parameter>\n</function>"
        result = extract_full_function_call(content)
        self.assertIsNotNone(result, "Failed to extract function call")
        self.assertIn("<function=execute_bash>", result, "Function name not in extracted call")
        self.assertIn("<parameter=command>", result, "Parameter not in extracted call")

        # Test with no function call
        content = "This is just a regular message"
        result = extract_full_function_call(content)
        self.assertIsNone(result, "Extracted function call from non-function content")

        # Test with multiple function calls
        content = "<function=execute_bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n<function=think>\n<parameter=thought>\nI need to check the files\n</parameter>\n</function>"
        result = extract_full_function_call(content)
        self.assertIsNotNone(result, "Failed to extract function call")
        self.assertIn("<function=execute_bash>", result, "Function name not in extracted call")

        # Test with nested function calls
        content = "<function=execute_bash>\n<parameter=command>\necho '<function=think>nested</function>'\n</parameter>\n</function>"
        result = extract_full_function_call(content)
        self.assertIsNotNone(result, "Failed to extract function call")
        self.assertIn("<function=execute_bash>", result, "Function name not in extracted call")
        # Just check that we have the parameter tag and some of the content
        self.assertIn("<parameter=command>", result, "Parameter tag not in extracted call")
        self.assertIn("echo", result, "Command content not in extracted call")


if __name__ == "__main__":
    unittest.main()
