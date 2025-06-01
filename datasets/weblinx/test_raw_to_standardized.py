#!/usr/bin/env python3
"""
Test script to verify the fixes in raw_to_standardized.py for weblinx dataset.
"""

import json
import sys
import unittest
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from datasets.weblinx.raw_to_standardized import sanitize_kwargs, sanitize_value


class TestWeblinxSanitization(unittest.TestCase):
    """Test the sanitization functions for weblinx dataset."""

    def test_sanitize_value_with_quotes(self):
        """Test sanitizing a value with quotes."""
        value = 'This has "quotes" in it'
        sanitized = sanitize_value(value)

        # Verify it can be used in JSON
        json_str = f'{{"value": "{sanitized}"}}'
        parsed = json.loads(json_str)
        self.assertEqual(parsed["value"], 'This has "quotes" in it')

    def test_sanitize_value_with_newlines(self):
        """Test sanitizing a value with newlines."""
        value = "This has\nnewlines\nin it"
        sanitized = sanitize_value(value)

        # Verify it can be used in JSON
        json_str = f'{{"value": "{sanitized}"}}'
        parsed = json.loads(json_str)
        self.assertEqual(parsed["value"], "This has\nnewlines\nin it")

    def test_sanitize_value_with_backslashes(self):
        """Test sanitizing a value with backslashes."""
        value = r"This has \ backslashes \ in it"
        sanitized = sanitize_value(value)

        # Verify it can be used in JSON
        json_str = f'{{"value": "{sanitized}"}}'
        parsed = json.loads(json_str)
        self.assertEqual(parsed["value"], r"This has \ backslashes \ in it")

    def test_sanitize_value_with_control_chars(self):
        """Test sanitizing a value with control characters."""
        value = "This has\tcontrol\rchars\x01in it"
        sanitized = sanitize_value(value)

        # Verify it can be used in JSON
        json_str = f'{{"value": "{sanitized}"}}'
        parsed = json.loads(json_str)
        # json.dumps preserves the control character \x01
        self.assertEqual(parsed["value"], "This has\tcontrol\rchars\x01in it")

    def test_sanitize_kwargs(self):
        """Test sanitizing a kwargs dictionary."""
        kwargs = {
            "xpath": "//*[@id='test\"id']",
            "value": "This has\nnewlines\nin it",
        }
        sanitized = sanitize_kwargs(kwargs)

        # Create a JSON string with the sanitized values
        json_str = f'{{"xpath": "{sanitized["xpath"]}", "value": "{sanitized["value"]}"}}'

        # Parse the JSON string
        try:
            parsed = json.loads(json_str)
            # If we get here, the JSON is valid
            self.assertTrue(True)
        except json.JSONDecodeError as e:
            self.fail(f"JSONDecodeError was raised: {e}\nJSON string: {json_str}")

    def test_api_action_in_std_to_sft(self):
        """Test that the ApiAction can be used in std_to_sft.py."""
        # This simulates the problematic code in std_to_sft.py
        kwargs = {
            "xpath": "//*[@id='test\"id']",
            "value": "This has\nnewlines\nin it",
        }
        sanitized = sanitize_kwargs(kwargs)

        # Convert to string as it would be in the code
        api_action_str = f"type(bid=123, xpath={json.dumps(sanitized['xpath'])}, value={json.dumps(sanitized['value'])})"

        # Sanitize the api_action_str itself for JSON
        api_action_str_sanitized = json.dumps(api_action_str)[1:-1]  # Use json.dumps directly

        # This is the line that was failing in std_to_sft.py
        try:
            json_str = (
                f'{{"name": "browser", "arguments": {{"code": "{api_action_str_sanitized}"}}}}'
            )
            call = json.loads(json_str)
            self.assertTrue(True)  # If we get here, the test passed
        except json.JSONDecodeError as e:
            self.fail(f"JSONDecodeError was raised: {e}\nJSON string: {json_str}")


if __name__ == "__main__":
    unittest.main()
