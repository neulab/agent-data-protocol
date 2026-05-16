"""
Tests for image observation handling in agents/openhands/std_to_sft.py
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

os.environ["MY_DATASET"] = "test_dataset"

from schema.observation.image import BoundingBox, ImageAnnotation, ImageObservation
from schema.observation.web import WebObservation

# ============================================================================
# Helper fixtures
# ============================================================================


def make_bounding_box(x=0.0, y=0.0, width=100.0, height=50.0):
    """Create a BoundingBox with default values."""
    return BoundingBox(x=x, y=y, width=width, height=height)


def make_annotation(
    element_type="button",
    text=None,
    content_description=None,
    clickable=None,
    editable=None,
    bounding_box=None,
):
    """Create an ImageAnnotation with configurable fields."""
    return ImageAnnotation(
        element_type=element_type,
        bounding_box=bounding_box or make_bounding_box(),
        text=text,
        content_description=content_description,
        clickable=clickable,
        editable=editable,
    )


def make_image_observation(
    content="/path/to/image.png",
    annotations=None,
    source="environment",
):
    """Create an ImageObservation."""
    return ImageObservation(
        class_="image_observation",
        content=content,
        annotations=annotations,
        source=source,
    )


def make_web_observation(
    html="<html><body>Test</body></html>",
    url="https://example.com",
    axtree="[1] button 'Submit'",
    image_observation=None,
    viewport_size=(1920, 1080),
):
    """Create a WebObservation with axtree pre-populated to skip generate_axtree."""
    return WebObservation(
        class_="web_observation",
        html=html,
        url=url,
        axtree=axtree,
        image_observation=image_observation,
        viewport_size=viewport_size,
    )


# ============================================================================
# Section 2: Nested ImageObservation in WebObservation
# ============================================================================


class TestWebObservationWithNestedImage:
    """Tests for WebObservation containing nested ImageObservation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Mock generate_axtree for WebObservation tests."""
        with patch("agents.openhands.std_to_sft.generate_axtree") as mock_axtree:
            mock_axtree.last_html = None
            mock_axtree.last_xtree = "[1] button 'Test'"
            yield mock_axtree

    def test_web_observation_no_image(self, setup):
        """WebObservation with no image_observation should not include VISUAL OBSERVATION."""
        from agents.openhands.std_to_sft import standardized_event_to_openhands_message

        web_obs = make_web_observation(image_observation=None)

        result = standardized_event_to_openhands_message(
            id="test_1",
            event=web_obs,
            previous_web_actions=[],
            is_web=True,
            api_env=None,
            api_sigs={},
            languages=[],
        )

        assert result["from"] == "human"
        assert result.get("_image_path") is None
        assert "VISUAL OBSERVATION" not in result["value"]
        assert "Elements detected" not in result["value"]
        assert "<image>" not in result["value"]

    def test_web_observation_with_image_no_annotations(self, setup):
        """WebObservation with ImageObservation but no annotations."""
        from agents.openhands.std_to_sft import standardized_event_to_openhands_message

        img_obs = make_image_observation(
            content="/path/to/screenshot.png",
            annotations=None,
        )
        web_obs = make_web_observation(image_observation=img_obs)

        result = standardized_event_to_openhands_message(
            id="test_2",
            event=web_obs,
            previous_web_actions=[],
            is_web=True,
            api_env=None,
            api_sigs={},
            languages=[],
        )

        assert result["from"] == "human"
        assert result["_image_path"] == "/path/to/screenshot.png"
        assert result["value"].endswith("\n\n---\nVISUAL OBSERVATION:\n<image>")
        assert "Elements detected" not in result["value"]

    def test_web_observation_with_image_full_annotation(self, setup):
        """WebObservation with ImageObservation containing full annotations."""
        from agents.openhands.std_to_sft import standardized_event_to_openhands_message

        annotation = make_annotation(
            element_type="button",
            text="Submit",
            clickable=True,
        )
        img_obs = make_image_observation(
            content="/path/to/screenshot.png",
            annotations=[annotation],
        )
        web_obs = make_web_observation(image_observation=img_obs)

        result = standardized_event_to_openhands_message(
            id="test_3",
            event=web_obs,
            previous_web_actions=[],
            is_web=True,
            api_env=None,
            api_sigs={},
            languages=[],
        )

        assert result["from"] == "human"
        assert result["_image_path"] == "/path/to/screenshot.png"
        assert result["value"].endswith(
            "\n\n---\nVISUAL OBSERVATION:\n<image>\nElements detected: Submit (button) [clickable]"
        )

    def test_web_observation_with_content_description_fallback(self, setup):
        """When text is None, content_description should be used."""
        from agents.openhands.std_to_sft import standardized_event_to_openhands_message

        annotation = make_annotation(
            element_type="icon",
            text=None,
            content_description="Settings icon",
            clickable=True,
        )
        img_obs = make_image_observation(
            content="/path/to/screenshot.png",
            annotations=[annotation],
        )
        web_obs = make_web_observation(image_observation=img_obs)

        result = standardized_event_to_openhands_message(
            id="test_fallback",
            event=web_obs,
            previous_web_actions=[],
            is_web=True,
            api_env=None,
            api_sigs={},
            languages=[],
        )

        assert result["value"].endswith(
            "\n\n---\nVISUAL OBSERVATION:\n<image>\nElements detected: Settings icon (icon) [clickable]"
        )

    def test_web_observation_with_multiple_annotations(self, setup):
        """Multiple annotations should be comma-separated."""
        from agents.openhands.std_to_sft import standardized_event_to_openhands_message

        annotations = [
            make_annotation(element_type="button", text="Submit", clickable=True),
            make_annotation(element_type="text_field", text="Username", editable=True),
        ]
        img_obs = make_image_observation(
            content="/path/to/screenshot.png",
            annotations=annotations,
        )
        web_obs = make_web_observation(image_observation=img_obs)

        result = standardized_event_to_openhands_message(
            id="test_multi",
            event=web_obs,
            previous_web_actions=[],
            is_web=True,
            api_env=None,
            api_sigs={},
            languages=[],
        )

        assert result["value"].endswith(
            "\n\n---\nVISUAL OBSERVATION:\n<image>\n"
            "Elements detected: Submit (button) [clickable], Username (text_field) [editable]"
        )


# ============================================================================
# Section 3: Standalone ImageObservation
# ============================================================================


class TestStandaloneImageObservation:
    """Tests for standalone ImageObservation handling."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Mock generate_axtree (not used for ImageObservation, but needed for import)."""
        with patch("agents.openhands.std_to_sft.generate_axtree"):
            yield

    def test_image_observation_no_annotations(self, setup):
        """ImageObservation with no annotations."""
        from agents.openhands.std_to_sft import standardized_event_to_openhands_message

        img_obs = make_image_observation(
            content="/path/to/image.png",
            annotations=None,
        )

        result = standardized_event_to_openhands_message(
            id="test_4",
            event=img_obs,
            previous_web_actions=[],
            is_web=False,
            api_env=None,
            api_sigs={},
            languages=[],
        )

        assert result == {
            "from": "observation",
            "value": "<image>",
            "_image_path": "/path/to/image.png",
        }

    def test_image_observation_with_annotations(self, setup):
        """ImageObservation with annotations."""
        from agents.openhands.std_to_sft import standardized_event_to_openhands_message

        annotation = make_annotation(
            element_type="button",
            text="Submit",
            clickable=True,
        )
        img_obs = make_image_observation(
            content="/path/to/image.png",
            annotations=[annotation],
        )

        result = standardized_event_to_openhands_message(
            id="test_5",
            event=img_obs,
            previous_web_actions=[],
            is_web=False,
            api_env=None,
            api_sigs={},
            languages=[],
        )

        assert result == {
            "from": "observation",
            "value": "<image>Elements detected: Submit (button) [clickable]",
            "_image_path": "/path/to/image.png",
        }

    def test_image_observation_with_clickable_and_editable(self, setup):
        """Both clickable and editable should appear in brackets."""
        from agents.openhands.std_to_sft import standardized_event_to_openhands_message

        annotation = make_annotation(
            element_type="text_field",
            text="Search",
            clickable=True,
            editable=True,
        )
        img_obs = make_image_observation(
            content="/path/to/image.png",
            annotations=[annotation],
        )

        result = standardized_event_to_openhands_message(
            id="test_both_attrs",
            event=img_obs,
            previous_web_actions=[],
            is_web=False,
            api_env=None,
            api_sigs={},
            languages=[],
        )

        assert result == {
            "from": "observation",
            "value": "<image>Elements detected: Search (text_field) [clickable, editable]",
            "_image_path": "/path/to/image.png",
        }


# ============================================================================
# Section 4: Image Path Collection in process_row
# ============================================================================


class TestImagePathCollection:
    """Tests for image path collection in process_row."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Mock generate_axtree and create test directory."""
        with patch("agents.openhands.std_to_sft.generate_axtree") as mock_axtree:
            mock_axtree.last_html = None
            mock_axtree.last_xtree = "[1] button 'Test'"
            mock_axtree.build_axtree = MagicMock(return_value="[1] button 'Test'")
            mock_axtree.get_bid = MagicMock(return_value='"test_bid"')
            yield mock_axtree

    def _make_trajectory_json(self, content):
        """Create a trajectory JSON line."""
        trajectory = {
            "id": "test_trajectory",
            "content": content,
            "details": {},
        }
        return json.dumps(trajectory)

    def test_process_row_no_images(self, setup):
        """Trajectory with no images should not have 'images' key."""
        from agents.openhands.std_to_sft import process_row
        from schema.observation.text import TextObservation

        text_obs = TextObservation(
            class_="text_observation",
            content="Hello, world!",
            source="user",
        )
        trajectory_json = self._make_trajectory_json([text_obs.model_dump()])

        result = process_row(
            line=trajectory_json,
            is_web=False,
            api_env=None,
            api_tool_description="",
            api_sigs={},
            export_for="explicit",
        )

        assert "images" not in result

    def test_process_row_with_image_observation(self, setup):
        """Trajectory with ImageObservation should collect image path."""
        from agents.openhands.std_to_sft import process_row
        from schema.observation.text import TextObservation

        text_obs = TextObservation(
            class_="text_observation",
            content="Hello, world!",
            source="user",
        )
        img_obs = make_image_observation(content="/path/to/image.png")

        trajectory_json = self._make_trajectory_json([text_obs.model_dump(), img_obs.model_dump()])

        result = process_row(
            line=trajectory_json,
            is_web=False,
            api_env=None,
            api_tool_description="",
            api_sigs={},
            export_for="explicit",
        )

        assert result["images"] == ["/path/to/image.png"]

    def test_process_row_web_with_nested_image(self, setup):
        """Trajectory with WebObservation containing nested image should collect path."""
        from agents.openhands.std_to_sft import process_row

        img_obs = make_image_observation(content="/path/to/screenshot.png")
        web_obs = make_web_observation(image_observation=img_obs)

        trajectory_json = self._make_trajectory_json([web_obs.model_dump()])

        result = process_row(
            line=trajectory_json,
            is_web=True,
            api_env=None,
            api_tool_description="",
            api_sigs={},
            export_for="explicit",
        )

        assert result["images"] == ["/path/to/screenshot.png"]
