import importlib.util
import inspect
import json
import os
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest
from pydantic import ValidationError

from schema.action.api import ApiAction
from schema.observation.image import ImageObservation
from schema.observation.text import TextObservation
from schema.tool_call_links import create_trajectory_with_tool_call_links
from schema.trajectory import Trajectory

TOOL_ACTION_CLASSES = {"api_action", "code_action"}
OBSERVATION_CLASSES = {"text_observation", "image_observation", "web_observation"}

DATASET_PATH = Path(__file__).parent.parent / "datasets"
NUMERIC_DETAIL_KEYS = {"reward", "score", "step"}
NUMERIC_DETAIL_KEY_PREFIXES = {"pred_passes_"}
NUMERIC_DETAIL_KEY_SUFFIXES = {
    "_correct",
    "_count",
    "_index",
    "_number",
    "_percentage",
    "_percent",
    "_rate",
    "_reward",
    "_score",
    "_success",
}


def is_numeric_detail_key(key):
    key = key.lower()
    return (
        key in NUMERIC_DETAIL_KEYS
        or any(key.startswith(prefix) for prefix in NUMERIC_DETAIL_KEY_PREFIXES)
        or any(key.endswith(suffix) for suffix in NUMERIC_DETAIL_KEY_SUFFIXES)
    )


def is_portable_or_external_image_reference(value):
    if value.startswith("data:"):
        return True
    if value.startswith("file://"):
        return False
    if "://" in value:
        return True

    windows_path = PureWindowsPath(value)
    return not (
        PurePosixPath(value).is_absolute() or windows_path.is_absolute() or bool(windows_path.drive)
    )


def get_sample_jsons(directory):
    # get DATASET_PATH/*/sample_std.json files
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        sample_path = os.path.join(subdir_path, "sample_std.json")
        if os.path.exists(sample_path):
            yield sample_path


def load_json(file_path):
    """Load JSON file, handling both indented and non-indented formats."""
    with open(file_path, "r") as file:
        return json.load(file)


def assert_tool_call_links_are_complete(sample, sample_path, sample_id):
    action_counts = Counter()
    observation_counts = Counter()
    content = sample.get("content", [])

    for content_id, item in enumerate(content):
        class_name = item.get("class_")
        tool_call_id = item.get("tool_call_id")

        if class_name in TOOL_ACTION_CLASSES:
            next_item = content[content_id + 1] if content_id + 1 < len(content) else {}
            if next_item.get("class_") in OBSERVATION_CLASSES:
                assert tool_call_id, (
                    f"Tool action followed by a result must include tool_call_id in "
                    f"{sample_path} sample {sample_id} content {content_id}"
                )

        if not tool_call_id:
            continue

        if class_name and class_name.endswith("_action"):
            action_counts[tool_call_id] += 1
        elif class_name in OBSERVATION_CLASSES:
            observation_counts[tool_call_id] += 1

    all_ids = sorted(set(action_counts) | set(observation_counts))
    invalid_ids = {
        tool_call_id: {
            "actions": action_counts[tool_call_id],
            "observations": observation_counts[tool_call_id],
        }
        for tool_call_id in all_ids
        if action_counts[tool_call_id] != 1 or observation_counts[tool_call_id] != 1
    }
    assert not invalid_ids, (
        f"Every tool_call_id must appear on exactly one action and one observation in "
        f"{sample_path} sample {sample_id}: {invalid_ids}"
    )


def test_numeric_detail_key_detection():
    assert is_numeric_detail_key("tool_call_count")
    assert is_numeric_detail_key("agent_percentage")
    assert is_numeric_detail_key("session_success")
    assert is_numeric_detail_key("rollout_number")
    assert is_numeric_detail_key("source_index")
    assert is_numeric_detail_key("reward")
    assert is_numeric_detail_key("gen_tests_correct")
    assert is_numeric_detail_key("pred_passes_gen_tests")

    assert not is_numeric_detail_key("source_id")
    assert not is_numeric_detail_key("timestamp")
    assert not is_numeric_detail_key("version")
    assert not is_numeric_detail_key("answer")


def test_image_reference_portability_detection():
    assert is_portable_or_external_image_reference("images/screenshot.png")
    assert is_portable_or_external_image_reference("s3://bucket/screenshot.png")
    assert is_portable_or_external_image_reference("data:image/png;base64,abc")

    assert not is_portable_or_external_image_reference("/Users/alice/screenshot.png")
    assert not is_portable_or_external_image_reference("C:\\Users\\alice\\screenshot.png")
    assert not is_portable_or_external_image_reference("file:///Users/alice/screenshot.png")


@pytest.mark.parametrize(
    "sample",
    [
        {
            "id": "extra-root",
            "content": [],
            "unexpected_root_field": True,
        },
        {
            "id": "extra-action",
            "content": [
                {
                    "class_": "message_action",
                    "content": "hello",
                    "unexpected_action_field": True,
                }
            ],
        },
        {
            "id": "extra-observation",
            "content": [
                {
                    "class_": "text_observation",
                    "content": "hello",
                    "source": "environment",
                    "unexpected_observation_field": True,
                }
            ],
        },
    ],
)
def test_standardized_schema_rejects_extra_fields(sample):
    with pytest.raises(ValidationError) as exc_info:
        Trajectory(**sample)

    # Pydantic v2 inlines branch errors from union fields into the top-level
    # error list, so extra_forbidden errors on nested Action/Observation models
    # surface here even though they originate inside the `content` union.
    assert any(error["type"] == "extra_forbidden" for error in exc_info.value.errors())


def test_standardized_schema_accepts_matched_tool_call_result():
    trajectory = Trajectory(
        id="matched-tool-result",
        content=[
            {
                "class_": "api_action",
                "tool_call_id": "call_000001",
                "function": "search",
                "kwargs": {"query": "agent data protocol"},
            },
            {
                "class_": "text_observation",
                "tool_call_id": "call_000001",
                "content": "Search result text",
                "source": "environment",
            },
        ],
    )

    action, observation = trajectory.content
    assert action.tool_call_id == "call_000001"
    assert observation.tool_call_id == "call_000001"


@pytest.mark.parametrize(
    ("action", "observation", "expected_tool_call_id"),
    [
        (
            ApiAction(function="search", kwargs={"query": "agent data protocol"}),
            TextObservation(content="Search result text", source="user"),
            "call_000001",
        ),
        (
            ApiAction(
                tool_call_id="call_from_action",
                function="search",
                kwargs={"query": "agent data protocol"},
            ),
            TextObservation(content="Search result text", source="environment"),
            "call_from_action",
        ),
        (
            ApiAction(function="search", kwargs={"query": "agent data protocol"}),
            TextObservation(
                tool_call_id="call_from_observation",
                content="Search result text",
                source="environment",
            ),
            "call_from_observation",
        ),
    ],
)
def test_raw_converter_helper_backfills_adjacent_tool_call_result(
    action, observation, expected_tool_call_id
):
    trajectory = create_trajectory_with_tool_call_links(
        id="backfilled-tool-result",
        content=[action, observation],
    )

    action, observation = trajectory.content
    assert action.tool_call_id == expected_tool_call_id
    assert observation.tool_call_id == expected_tool_call_id
    assert observation.source == "environment"


def test_standardized_schema_rejects_unmatched_tool_result():
    with pytest.raises(ValidationError, match="does not match any preceding Action"):
        Trajectory(
            id="unmatched-tool-result",
            content=[
                {
                    "class_": "text_observation",
                    "tool_call_id": "call_missing",
                    "content": "orphaned result",
                    "source": "environment",
                }
            ],
        )


def test_standardized_schema_rejects_unmatched_tool_call():
    with pytest.raises(ValidationError, match="does not have a matching Observation"):
        Trajectory(
            id="unmatched-tool-call",
            content=[
                {
                    "class_": "api_action",
                    "tool_call_id": "call_without_result",
                    "function": "search",
                    "kwargs": {"query": "agent data protocol"},
                }
            ],
        )


def test_standardized_schema_rejects_duplicate_tool_call_ids():
    with pytest.raises(ValidationError, match="Duplicate Action.tool_call_id"):
        Trajectory(
            id="duplicate-tool-call-id",
            content=[
                {
                    "class_": "api_action",
                    "tool_call_id": "call_duplicate",
                    "function": "search",
                    "kwargs": {"query": "first"},
                },
                {
                    "class_": "api_action",
                    "tool_call_id": "call_duplicate",
                    "function": "search",
                    "kwargs": {"query": "second"},
                },
            ],
        )


def test_standardized_schema_rejects_duplicate_tool_results():
    with pytest.raises(ValidationError, match="Duplicate observation result"):
        Trajectory(
            id="duplicate-tool-result",
            content=[
                {
                    "class_": "api_action",
                    "tool_call_id": "call_000001",
                    "function": "search",
                    "kwargs": {"query": "agent data protocol"},
                },
                {
                    "class_": "text_observation",
                    "tool_call_id": "call_000001",
                    "content": "first result",
                    "source": "environment",
                },
                {
                    "class_": "text_observation",
                    "tool_call_id": "call_000001",
                    "content": "second result",
                    "source": "environment",
                },
            ],
        )


def test_standardized_schema_rejects_user_source_tool_result():
    with pytest.raises(ValidationError, match="must not use source='user'"):
        Trajectory(
            id="user-source-tool-result",
            content=[
                {
                    "class_": "api_action",
                    "tool_call_id": "call_000001",
                    "function": "search",
                    "kwargs": {"query": "agent data protocol"},
                },
                {
                    "class_": "text_observation",
                    "tool_call_id": "call_000001",
                    "content": "Search result text",
                    "source": "user",
                },
            ],
        )


@pytest.mark.parametrize("sample_path", get_sample_jsons(DATASET_PATH))
def test_sample_standardized_against_schema(sample_path):
    samples = load_json(sample_path)
    assert isinstance(samples, list), "sample_std.json should be a list"
    assert len(samples) > 0, "sample_std.json should have at least one sample"

    # dynamically load api.py in the same directory as sample_std.json
    dataset_api = None
    api_function_names = None

    def load_dataset_api():
        nonlocal dataset_api, api_function_names
        if dataset_api is None:
            api_path = os.path.join(os.path.dirname(sample_path), "api.py")
            assert os.path.exists(api_path)
            spec = importlib.util.spec_from_file_location("dataset_api", api_path)
            dataset_api = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dataset_api)
            api_function_names = {
                name for name, _ in inspect.getmembers(dataset_api, inspect.isfunction)
            }
        return dataset_api, api_function_names

    for sample_id, sample in enumerate(samples):
        assert_tool_call_links_are_complete(sample, sample_path, sample_id)
        try:
            traj = Trajectory(**sample)
            assert "available_apis" not in traj.details, (
                f"available_apis must be a top-level Trajectory field, not details metadata, "
                f"in {sample_path} sample {sample_id}"
            )
            assert "system_prompt" not in traj.details, (
                f"system_prompt should not be stored in details metadata in "
                f"{sample_path} sample {sample_id}"
            )
            stringified_numeric_details = {
                key: value
                for key, value in traj.details.items()
                if is_numeric_detail_key(key) and isinstance(value, str)
            }
            assert not stringified_numeric_details, (
                f"Numeric details must be stored as native JSON numbers in {sample_path} "
                f"sample {sample_id}: {stringified_numeric_details}"
            )
            if traj.available_apis is not None:
                available_apis = traj.available_apis
                _, api_function_names = load_dataset_api()
                missing_available_apis = sorted(set(available_apis) - api_function_names)
                assert not missing_available_apis, (
                    f"available_apis contains functions not found in api.py in "
                    f"{os.path.dirname(sample_path)}: {missing_available_apis}"
                )
                used_apis = {
                    content.function for content in traj.content if isinstance(content, ApiAction)
                }
                missing_used_apis = sorted(used_apis - set(available_apis))
                assert not missing_used_apis, (
                    f"ApiAction functions are missing from available_apis in {sample_path} "
                    f"sample {sample_id}: {missing_used_apis}"
                )

            for content_id, content in enumerate(traj.content):
                print(f"{sample_id=}, {content_id=}, {type(content)=}")
                if isinstance(content, ImageObservation):
                    assert is_portable_or_external_image_reference(content.content), (
                        f"ImageObservation.content must be a portable relative path or "
                        f"external reference in {sample_path} sample {sample_id} "
                        f"content {content_id}: {content.content}"
                    )
                if isinstance(content, ApiAction):
                    # Make sure that content.function exists in api.py
                    dataset_api, _ = load_dataset_api()
                    assert hasattr(dataset_api, content.function), (
                        f"{content.function} not found in api.py in {os.path.dirname(sample_path)}"
                    )
                    # Validate content.kwargs against the function signature
                    function = getattr(dataset_api, content.function)
                    function(**content.kwargs)

        except ValidationError as e:
            pytest.fail(f"Validation failed for {sample_path}: {str(e)}")
