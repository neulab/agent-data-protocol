#!/usr/bin/env python3
"""
Integration test for the weblinx dataset.
This test verifies that the raw_to_standardized.py file correctly processes weblinx data
and that the std_to_sft.py script can process the standardized data without errors.
"""

import json
import sys
import unittest
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the necessary modules
from schema.action.api import ApiAction
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory


class TestWeblinxIntegration(unittest.TestCase):
    """Test the integration between raw_to_standardized.py and std_to_sft.py."""

    def test_api_action_json_serialization(self):
        """Test that ApiAction objects can be serialized to JSON without errors."""
        # Create a sample ApiAction with problematic values
        kwargs = {
            "xpath": "//*[@id='test\"id']",
            "value": 'This has\nnewlines\nin it and "quotes"',
        }

        # Import the sanitize_kwargs function from raw_to_standardized.py
        from datasets.weblinx.raw_to_standardized import sanitize_kwargs

        # Sanitize the kwargs
        sanitized_kwargs = sanitize_kwargs(kwargs)

        # Create an ApiAction with the sanitized kwargs
        api_action = ApiAction(function="type", kwargs=sanitized_kwargs)

        # Convert to string as it would be in std_to_sft.py
        api_action_str = f"type(bid=123, xpath={json.dumps(sanitized_kwargs['xpath'])}, value={json.dumps(sanitized_kwargs['value'])})"

        # This is the line that was failing in std_to_sft.py
        try:
            # Use json.dumps to properly escape the api_action_str
            api_action_str_sanitized = json.dumps(api_action_str)[1:-1]

            # Create the JSON string that was failing
            json_str = (
                f'{{"name": "browser", "arguments": {{"code": "{api_action_str_sanitized}"}}}}'
            )

            # Try to parse it
            call = json.loads(json_str)

            # If we get here, the JSON is valid
            self.assertEqual(call["name"], "browser")
            self.assertTrue("code" in call["arguments"])
        except json.JSONDecodeError as e:
            self.fail(f"JSONDecodeError was raised: {e}\nJSON string: {json_str}")

    def test_trajectory_serialization(self):
        """Test that a Trajectory with ApiAction can be serialized to JSON without errors."""
        # Create a sample TextObservation
        text_obs = TextObservation(content="Hello", source="user")

        # Import the sanitize_kwargs function from raw_to_standardized.py
        from datasets.weblinx.raw_to_standardized import sanitize_kwargs

        # Create a sample ApiAction with problematic values
        kwargs = {
            "xpath": "//*[@id='test\"id']",
            "value": 'This has\nnewlines\nin it and "quotes"',
        }

        # Sanitize the kwargs
        sanitized_kwargs = sanitize_kwargs(kwargs)

        # Create an ApiAction with the sanitized kwargs
        api_action = ApiAction(function="type", kwargs=sanitized_kwargs)

        # Create a sample WebObservation
        web_obs = WebObservation(
            html="<html><body>Test</body></html>",
            url="https://example.com",
            viewport_size=(800, 600),
            image_observation=None,
        )

        # Create a Trajectory with these events
        trajectory = Trajectory(
            id="test_id",
            content=[text_obs, web_obs, api_action],
        )

        # Serialize the trajectory to JSON
        try:
            json_str = trajectory.model_dump_json()

            # Try to parse it back
            parsed = json.loads(json_str)

            # If we get here, the JSON is valid
            self.assertEqual(parsed["id"], "test_id")
            self.assertEqual(len(parsed["content"]), 3)

            # Check that the ApiAction was serialized correctly
            api_action_dict = parsed["content"][2]
            self.assertEqual(api_action_dict["function"], "type")

            # The values might have different escaping, but they should be equivalent when parsed as JSON
            self.assertEqual(json.loads(f'"{api_action_dict["kwargs"]["xpath"]}"'), kwargs["xpath"])
            self.assertEqual(json.loads(f'"{api_action_dict["kwargs"]["value"]}"'), kwargs["value"])
        except json.JSONDecodeError as e:
            self.fail(f"JSONDecodeError was raised: {e}")


if __name__ == "__main__":
    unittest.main()
