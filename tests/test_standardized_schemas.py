import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest
from pydantic import ValidationError

from schema.atif import ATIFTrajectory
from schema.dataset_metadata import (
    custom_tool_map,
    custom_tool_names,
    is_browser_api_action,
    load_dataset_metadata,
)

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
BUILT_IN_TOOL_SCHEMAS = {
    "execute_bash": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "is_input": {"type": "boolean"},
            "reset": {"type": "boolean"},
            "security_risk": {"type": "string"},
            "summary": {"type": "string"},
            "timeout": {"type": "number"},
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    "terminal": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "is_input": {"type": "boolean"},
            "reset": {"type": "boolean"},
            "security_risk": {"type": "string"},
            "summary": {"type": "string"},
            "timeout": {"type": "number"},
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    "execute_ipython_cell": {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
        "additionalProperties": False,
    },
    "python": {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
        "additionalProperties": False,
    },
    "finish": {
        "type": "object",
        "properties": {
            "message": {"type": "string"},
            "task_completed": {"anyOf": [{"type": "boolean"}, {"type": "string"}]},
        },
        "required": ["message", "task_completed"],
        "additionalProperties": False,
    },
    "stop": {
        "type": "object",
        "properties": {"message": {"type": "string"}, "output": {"type": "string"}},
        "additionalProperties": False,
    },
    "submit": {"type": "object", "properties": {}, "additionalProperties": False},
    "str_replace_editor": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "path": {"type": "string"},
            "file_text": {"type": "string"},
            "old_str": {"type": "string"},
            "new_str": {"type": "string"},
            "insert_line": {"type": "integer"},
            "summary": {"type": "string"},
            "view_range": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["command", "path"],
        "additionalProperties": False,
    },
    "file_editor": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "path": {"type": "string"},
            "file_text": {"type": "string"},
            "old_str": {"type": "string"},
            "new_str": {"type": "string"},
            "insert_line": {"type": "integer"},
            "summary": {"type": "string"},
            "view_range": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["command", "path"],
        "additionalProperties": False,
    },
    "think": {
        "type": "object",
        "properties": {"thought": {"type": "string"}},
        "required": ["thought"],
        "additionalProperties": False,
    },
    "task_tracker": {"type": "object", "properties": {}, "additionalProperties": True},
}


def validate_schema_value(schema, value, path):
    if not schema:
        return []
    if "anyOf" in schema:
        option_errors = [validate_schema_value(option, value, path) for option in schema["anyOf"]]
        if any(not errors for errors in option_errors):
            return []
        return [f"{path} did not match anyOf: {option_errors[0]}"]
    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(value, dict):
            return [f"{path} must be an object"]
        errors = []
        properties = schema.get("properties", {}) or {}
        for key in schema.get("required", []) or []:
            if key not in value:
                errors.append(f"{path}.{key} is required")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                errors.append(f"{path} has unsupported properties {extra}")
        for key, item in value.items():
            if key in properties:
                errors.extend(validate_schema_value(properties[key], item, f"{path}.{key}"))
        return errors
    if schema_type == "array" and not isinstance(value, list):
        return [f"{path} must be an array"]
    if schema_type == "boolean" and not isinstance(value, bool):
        return [f"{path} must be a boolean"]
    if schema_type == "integer" and not (isinstance(value, int) and not isinstance(value, bool)):
        return [f"{path} must be an integer"]
    if schema_type == "number" and not (
        isinstance(value, (int, float)) and not isinstance(value, bool)
    ):
        return [f"{path} must be a number"]
    if schema_type == "string" and not isinstance(value, str):
        return [f"{path} must be a string"]
    enum = schema.get("enum")
    if enum is not None and value not in enum:
        return [f"{path} must be one of {enum}"]
    return []


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
    for subdir in os.listdir(directory):
        sample_path = os.path.join(directory, subdir, "sample_std.json")
        if os.path.exists(sample_path):
            yield sample_path


def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


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


def test_standardized_atif_schema_rejects_extra_fields():
    with pytest.raises(ValidationError) as exc_info:
        ATIFTrajectory(
            trajectory_id="extra-step",
            steps=[{"step_id": 1, "source": "user", "message": "hello", "extra_field": True}],
        )
    assert any(error["type"] == "extra_forbidden" for error in exc_info.value.errors())


@pytest.mark.parametrize("sample_path", get_sample_jsons(DATASET_PATH))
def test_sample_standardized_atif_against_schema(sample_path):
    samples = load_json(sample_path)
    assert isinstance(samples, list), "sample_std.json should be a list"
    assert len(samples) > 0, "sample_std.json should have at least one sample"

    dataset_name = Path(sample_path).parent.name
    metadata = load_dataset_metadata(dataset_name, required=True)
    api_function_names = custom_tool_names(metadata)
    api_function_specs = custom_tool_map(metadata)
    built_in_api_names = set(BUILT_IN_TOOL_SCHEMAS) | {"execute_code"}

    for sample_id, sample in enumerate(samples):
        try:
            traj = ATIFTrajectory(**sample)
            extra = traj.extra or {}
            assert "available_apis" not in extra.get("adp_details", {}), (
                f"available_apis must be kept as ATIF metadata, not details metadata, "
                f"in {sample_path} sample {sample_id}"
            )
            assert "system_prompt" not in extra.get("adp_details", {}), (
                f"system_prompt should not be stored in details metadata in "
                f"{sample_path} sample {sample_id}"
            )
            details = extra.get("adp_details", {})
            stringified_numeric_details = {
                key: value
                for key, value in details.items()
                if is_numeric_detail_key(key) and isinstance(value, str)
            }
            assert not stringified_numeric_details, (
                f"Numeric details must be stored as native JSON numbers in {sample_path} "
                f"sample {sample_id}: {stringified_numeric_details}"
            )
            available_apis = extra.get("adp_available_apis")
            if available_apis is not None:
                unsupported_available_apis = sorted(
                    name
                    for name in set(available_apis)
                    if name not in api_function_names
                    and name not in built_in_api_names
                    and not is_browser_api_action(name, browser_context=metadata.browser_enabled)
                )
                assert not unsupported_available_apis, (
                    f"available_apis contains functions not found in metadata.json in "
                    f"{os.path.dirname(sample_path)}: {unsupported_available_apis}"
                )
                used_apis = {
                    tool_call.function_name
                    for step in traj.steps
                    for tool_call in (step.tool_calls or [])
                    if tool_call.function_name
                    not in {"execute_bash", "execute_ipython_cell", "terminal", "python"}
                }
                missing_used_apis = sorted(used_apis - set(available_apis))
                assert not missing_used_apis, (
                    f"ATIF tool calls are missing from available_apis in {sample_path} "
                    f"sample {sample_id}: {missing_used_apis}"
                )

            for step_id, step in enumerate(traj.steps):
                step_contents = [step.message]
                if step.observation:
                    step_contents.extend(result.content for result in step.observation.results)
                for content_id, content in enumerate(step_contents):
                    image_paths = []
                    if not isinstance(content, str):
                        image_paths = [
                            part.source.path
                            for part in content
                            if part.type == "image" and part.source is not None
                        ]
                    for image_path in image_paths:
                        assert is_portable_or_external_image_reference(image_path), (
                            f"ImageObservation.content must be a portable relative path or "
                            f"external reference in {sample_path} sample {sample_id} "
                            f"step {step_id} content {content_id}: {image_path}"
                        )
                for tool_call in step.tool_calls or []:
                    supported_by_metadata = (
                        tool_call.function_name in api_function_names
                        or tool_call.function_name in built_in_api_names
                        or is_browser_api_action(
                            tool_call.function_name,
                            tool_call.arguments,
                            browser_context=metadata.browser_enabled,
                        )
                    )
                    assert supported_by_metadata, (
                        f"{tool_call.function_name} not found in metadata.json in "
                        f"{os.path.dirname(sample_path)}"
                    )
                    if tool_call.function_name in {
                        "execute_bash",
                        "execute_ipython_cell",
                        "terminal",
                        "python",
                    }:
                        expected_language = (
                            "bash"
                            if tool_call.function_name in {"execute_bash", "terminal"}
                            else "python"
                        )
                        assert expected_language in metadata.code_enabled, (
                            f"{tool_call.function_name} is used in {sample_path} sample "
                            f"{sample_id}, but metadata.json code_enabled does not include "
                            f"{expected_language!r}"
                        )
                    if tool_call.function_name in {"str_replace_editor", "file_editor"}:
                        assert metadata.file_editor_enabled, (
                            f"{tool_call.function_name} is used in {sample_path} sample {sample_id}, "
                            "but metadata.json file_editor_enabled is false"
                        )
                    if tool_call.function_name in api_function_specs:
                        argument_schema = api_function_specs[
                            tool_call.function_name
                        ].function.parameters
                    elif tool_call.function_name in BUILT_IN_TOOL_SCHEMAS:
                        argument_schema = BUILT_IN_TOOL_SCHEMAS[tool_call.function_name]
                    else:
                        argument_schema = None
                    if argument_schema is not None:
                        argument_errors = validate_schema_value(
                            argument_schema,
                            tool_call.arguments,
                            f"{tool_call.function_name}.arguments",
                        )
                        assert not argument_errors, (
                            f"Tool arguments do not match metadata/schema in {sample_path} "
                            f"sample {sample_id} step {step_id}: {argument_errors}"
                        )
        except ValidationError as e:
            pytest.fail(f"Validation failed for {sample_path}: {str(e)}")
