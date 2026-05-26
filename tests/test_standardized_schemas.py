import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest
from pydantic import ValidationError

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.dataset_metadata import (
    DatasetMetadata,
    custom_tool_names,
    infer_metadata_usage,
    is_browser_api_action,
    load_dataset_metadata,
    validate_trajectory_metadata,
)
from schema.observation.image import ImageObservation
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.tool_call_links import create_trajectory_with_tool_call_links
from schema.trajectory import Trajectory

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


def validate_kwargs_against_openai_tool(
    content: ApiAction,
    metadata: DatasetMetadata,
    sample_path: str,
    sample_id: int,
    content_id: int,
) -> None:
    tools_by_name = {tool.function.name: tool for tool in metadata.custom_tools}
    tool = tools_by_name.get(content.function)
    if tool is None:
        return
    parameters = tool.function.parameters or {}
    properties = parameters.get("properties") or {}
    required = set(parameters.get("required") or [])
    provided = set(content.kwargs)
    missing_required = sorted(required - provided)
    assert not missing_required, (
        f"ApiAction {content.function!r} is missing required metadata.json "
        f"arguments in {sample_path} sample {sample_id} content {content_id}: "
        f"{missing_required}"
    )
    if parameters.get("additionalProperties") is not True:
        unexpected = sorted(provided - set(properties))
        assert not unexpected, (
            f"ApiAction {content.function!r} has kwargs not declared by metadata.json "
            f"in {sample_path} sample {sample_id} content {content_id}: {unexpected}"
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
            CodeAction(language="bash", content="pwd", description=None),
            TextObservation(content="/workspace/project", source="environment"),
            "call_000001",
        ),
        (
            ApiAction(function="screenshot", kwargs={}),
            ImageObservation(content="screen.png", source="user", annotations=None),
            "call_000001",
        ),
        (
            ApiAction(function="observe", kwargs={}),
            WebObservation(
                html="<html></html>",
                axtree=None,
                url="https://example.com",
                image_observation=None,
                viewport_size=None,
            ),
            "call_000001",
        ),
        (
            ApiAction(
                tool_call_id="call_from_both",
                function="search",
                kwargs={"query": "agent data protocol"},
            ),
            TextObservation.model_construct(
                tool_call_id="call_from_both",
                content="Search result text",
                source="user",
            ),
            "call_from_both",
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
    if isinstance(observation, (TextObservation, ImageObservation)):
        assert observation.source == "environment"


def test_raw_converter_helper_infers_available_code_languages():
    trajectory = create_trajectory_with_tool_call_links(
        id="code-languages",
        content=[
            CodeAction(language="bash", content="pwd", description=None),
            TextObservation(content="/workspace/project", source="environment"),
            CodeAction(language="python", content="print('ok')", description=None),
            TextObservation(content="ok", source="environment"),
        ],
    )

    assert trajectory.available_code_languages == ["bash", "python"]


def test_standardized_schema_rejects_missing_available_code_language():
    with pytest.raises(
        ValidationError,
        match="CodeAction languages are missing from available_code_languages",
    ):
        Trajectory(
            id="missing-code-language",
            available_code_languages=["python"],
            content=[
                {
                    "class_": "code_action",
                    "language": "bash",
                    "content": "pwd",
                    "description": None,
                }
            ],
        )


def test_standardized_schema_rejects_unused_available_code_language():
    with pytest.raises(
        ValidationError,
        match="available_code_languages contains languages not used by CodeAction",
    ):
        Trajectory(
            id="unused-code-language",
            available_code_languages=["bash", "python"],
            content=[
                {
                    "class_": "code_action",
                    "language": "bash",
                    "content": "pwd",
                    "description": None,
                }
            ],
        )


def test_standardized_schema_rejects_tool_call_id_on_message_action():
    with pytest.raises(ValidationError, match="MessageAction.tool_call_id"):
        Trajectory(
            id="message-action-tool-call-id",
            content=[
                {
                    "class_": "message_action",
                    "tool_call_id": "call_message",
                    "content": "Done.",
                }
            ],
        )


def test_standardized_schema_rejects_missing_tool_call_id_for_tool_result():
    with pytest.raises(ValidationError, match="does not include tool_call_id"):
        Trajectory(
            id="missing-tool-call-id",
            content=[
                {
                    "class_": "api_action",
                    "function": "search",
                    "kwargs": {"query": "agent data protocol"},
                },
                {
                    "class_": "text_observation",
                    "content": "Search result text",
                    "source": "environment",
                },
            ],
        )


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


@pytest.mark.parametrize(
    "content",
    [
        [
            {
                "class_": "api_action",
                "tool_call_id": "call_without_result",
                "function": "search",
                "kwargs": {"query": "agent data protocol"},
            }
        ],
        [
            {
                "class_": "api_action",
                "tool_call_id": "call_without_result",
                "function": "search",
                "kwargs": {"query": "agent data protocol"},
            },
            {
                "class_": "text_observation",
                "content": "Search result text",
                "source": "environment",
            },
        ],
    ],
)
def test_standardized_schema_rejects_unmatched_tool_call(content):
    with pytest.raises(ValidationError, match="does not have a matching Observation"):
        Trajectory(id="unmatched-tool-call", content=content)


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

    dataset_name = Path(sample_path).parent.name
    metadata_path = Path(sample_path).with_name("metadata.json")
    assert metadata_path.exists(), f"metadata.json not found for {dataset_name}"
    metadata = load_dataset_metadata(dataset_name, required=True)
    metadata_custom_tool_names = custom_tool_names(metadata)
    trajectories = []

    for sample_id, sample in enumerate(samples):
        try:
            traj = Trajectory(**sample)
            trajectories.append(traj)
            validate_trajectory_metadata(traj, metadata, dataset_name=dataset_name)
            assert "available_custom_tools" not in traj.details, (
                f"available_custom_tools must be a top-level Trajectory field, not "
                f"details metadata, in {sample_path} sample {sample_id}"
            )
            assert "available_code_languages" not in traj.details, (
                f"available_code_languages must be a top-level Trajectory field, "
                f"not details metadata, in {sample_path} sample {sample_id}"
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
            if traj.available_custom_tools is not None:
                missing_available_tools = sorted(
                    set(traj.available_custom_tools) - metadata_custom_tool_names
                )
                assert not missing_available_tools, (
                    f"available_custom_tools contains tools not found in metadata.json "
                    f"in {os.path.dirname(sample_path)}: {missing_available_tools}"
                )
                used_custom_tools = {
                    content.function
                    for content in traj.content
                    if isinstance(content, ApiAction)
                    and not is_browser_api_action(
                        content.function,
                        content.kwargs,
                        browser_context=metadata.browser_enabled,
                    )
                }
                missing_used_tools = sorted(
                    used_custom_tools - set(traj.available_custom_tools)
                )
                assert not missing_used_tools, (
                    f"ApiAction functions are missing from available_custom_tools in "
                    f"{sample_path} sample {sample_id}: {missing_used_tools}"
                )

            used_code_languages = {
                content.language for content in traj.content if isinstance(content, CodeAction)
            }
            if used_code_languages:
                assert set(traj.available_code_languages or []) == used_code_languages, (
                    f"available_code_languages must exactly match CodeAction languages in "
                    f"{sample_path} sample {sample_id}: expected "
                    f"{sorted(used_code_languages)}, got {traj.available_code_languages}"
                )
            else:
                assert traj.available_code_languages is None, (
                    f"available_code_languages should be omitted when there are no "
                    f"CodeAction entries in {sample_path} sample {sample_id}"
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
                    if not is_browser_api_action(
                        content.function,
                        content.kwargs,
                        browser_context=metadata.browser_enabled,
                    ):
                        assert content.function in metadata_custom_tool_names, (
                            f"{content.function} not found in metadata.json custom_tools "
                            f"in {os.path.dirname(sample_path)}"
                        )
                    validate_kwargs_against_openai_tool(
                        content, metadata, sample_path, sample_id, content_id
                    )

        except ValidationError as e:
            pytest.fail(f"Validation failed for {sample_path}: {str(e)}")

    used_code_languages, browser_enabled, _ = infer_metadata_usage(trajectories)
    assert metadata.code_enabled == sorted(used_code_languages), (
        f"metadata.json code_enabled must exactly match CodeAction languages in "
        f"{sample_path}: expected {sorted(used_code_languages)}, got {metadata.code_enabled}"
    )
    assert metadata.browser_enabled is browser_enabled, (
        f"metadata.json browser_enabled must match browser usage in {sample_path}: "
        f"expected {browser_enabled}, got {metadata.browser_enabled}"
    )
