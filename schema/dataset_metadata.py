import inspect
import json
import re
import textwrap
from pathlib import Path
from typing import Any, Literal, get_args, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.observation.web import WebObservation

DATASET_ROOT = Path(__file__).parent.parent / "datasets"

BROWSER_ACTION_NAMES = {
    "back",
    "click",
    "clear",
    "dblclick",
    "drag_and_drop",
    "fill",
    "focus",
    "go_back",
    "go_forward",
    "goto",
    "hover",
    "noop",
    "press",
    "scroll",
    "select_option",
    "type",
    "upload_file",
}
BROWSER_KWARG_NAMES = {
    "bid",
    "delta_x",
    "delta_y",
    "dx",
    "dy",
    "element",
    "element_id",
    "id",
    "index",
    "url",
    "xpath",
}


class OpenAIFunctionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


class OpenAIToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["function"] = "function"
    function: OpenAIFunctionSpec


class DatasetMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    custom_tools: list[OpenAIToolSpec] = Field(default_factory=list)
    code_enabled: list[str] = Field(default_factory=list)
    browser_enabled: bool = False

    @field_validator("code_enabled")
    def validate_code_enabled(cls, value):
        valid_languages = set(get_args(CodeAction.model_fields["language"].annotation))
        invalid_languages = sorted(set(value) - valid_languages)
        if invalid_languages:
            raise ValueError(
                f"code_enabled contains unsupported CodeAction languages: {invalid_languages}"
            )
        return sorted(value)

    @model_validator(mode="after")
    def validate_unique_custom_tool_names(self):
        names = [tool.function.name for tool in self.custom_tools]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"custom_tools contains duplicate tool names: {duplicates}")
        return self


def metadata_path_for_dataset(dataset_name: str, dataset_root: Path = DATASET_ROOT) -> Path:
    return dataset_root / dataset_name / "metadata.json"


def load_dataset_metadata(
    dataset_name: str | None,
    *,
    required: bool = False,
    dataset_root: Path = DATASET_ROOT,
) -> DatasetMetadata:
    if not dataset_name:
        if required:
            raise ValueError("MY_DATASET must be set to load dataset metadata")
        return DatasetMetadata()
    path = metadata_path_for_dataset(dataset_name, dataset_root)
    if not path.exists():
        if required:
            raise ValueError(f"metadata.json not found for dataset {dataset_name!r}: {path}")
        return DatasetMetadata()
    return DatasetMetadata(**json.loads(path.read_text()))


def custom_tool_names(metadata: DatasetMetadata) -> set[str]:
    return {tool.function.name for tool in metadata.custom_tools}


def custom_tool_map(metadata: DatasetMetadata) -> dict[str, dict[str, Any]]:
    return {
        tool.function.name: tool.model_dump(exclude_none=True) for tool in metadata.custom_tools
    }


def is_browser_api_action(
    function_name: str,
    kwargs: dict[str, Any] | None = None,
    *,
    browser_context: bool = False,
) -> bool:
    if function_name not in BROWSER_ACTION_NAMES:
        return False
    if browser_context:
        return True
    if kwargs is None:
        return False
    if function_name == "goto":
        return "url" in kwargs
    if function_name in {"back", "go_back", "go_forward", "noop"}:
        return False
    return bool(BROWSER_KWARG_NAMES & set(kwargs))


def split_docstring(docstring: str) -> tuple[str, dict[str, str]]:
    if not docstring:
        return "", {}

    lines = textwrap.dedent(docstring).strip().split("\n")
    summary_lines: list[str] = []
    arg_lines: list[str] = []
    in_args = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("Args:", "Arguments:", "Parameters:")):
            in_args = True
            continue
        if stripped in {"----", "-------"}:
            continue
        (arg_lines if in_args else summary_lines).append(line)

    summary = "\n".join(summary_lines).strip()
    arg_descriptions: dict[str, str] = {}
    current_name: str | None = None
    buffer: list[str] = []

    def flush_current() -> None:
        nonlocal buffer, current_name
        if current_name and buffer:
            arg_descriptions[current_name] = " ".join(part.strip() for part in buffer).strip()
        buffer = []

    for line in arg_lines:
        stripped = line.strip()
        if not stripped:
            flush_current()
            continue
        if ":" in stripped and not stripped.startswith(("-", "*")):
            before, after = stripped.split(":", 1)
            name = before.split("(", 1)[0].strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                flush_current()
                current_name = name
                if after.strip():
                    buffer.append(after.strip())
                continue
        buffer.append(stripped)
    flush_current()
    return summary, arg_descriptions


def json_schema_from_type(py_type: Any) -> dict[str, Any]:
    try:
        return TypeAdapter(py_type).json_schema()
    except Exception:
        return {"type": "string"}


def openai_tool_from_function(function: Any) -> dict[str, Any]:
    signature = inspect.signature(function)
    try:
        hints = get_type_hints(function, globalns=function.__globals__, include_extras=True)
    except Exception:
        hints = {}
    summary, arg_descriptions = split_docstring(inspect.getdoc(function) or "")
    if not summary:
        summary = f"Dataset-specific tool `{function.__name__}`."

    properties: dict[str, Any] = {}
    required: list[str] = []
    for parameter_name, parameter in signature.parameters.items():
        if parameter_name == "self":
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        schema = json_schema_from_type(hints.get(parameter_name, str))
        description = arg_descriptions.get(parameter_name)
        if description:
            schema["description"] = description
        properties[parameter_name] = schema
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter_name)

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        parameters["required"] = required

    return {
        "type": "function",
        "function": {
            "name": function.__name__,
            "description": summary,
            "parameters": parameters,
        },
    }


def openai_tool_signature(tool: OpenAIToolSpec) -> tuple[list[str], list[str], str]:
    parameters = tool.function.parameters or {}
    properties = parameters.get("properties") or {}
    required = list(parameters.get("required") or [])
    optional = [name for name in properties if name not in required]
    return required, optional, _format_signature(tool.function.name, required, optional)


def _format_signature(name: str, required: list[str], optional: list[str]) -> str:
    args = required + [f"{arg}=None" for arg in optional]
    return f"{name}({', '.join(args)})"


def infer_metadata_usage(trajectories: list[Any]) -> tuple[set[str], bool, set[str]]:
    code_languages: set[str] = set()
    has_web_observation = any(
        isinstance(item, WebObservation) for trajectory in trajectories for item in trajectory.content
    )
    browser_enabled = has_web_observation
    api_functions: set[str] = set()
    for trajectory in trajectories:
        for item in trajectory.content:
            if isinstance(item, CodeAction):
                code_languages.add(item.language)
            elif isinstance(item, ApiAction):
                api_functions.add(item.function)
                if is_browser_api_action(
                    item.function,
                    item.kwargs,
                    browser_context=has_web_observation,
                ):
                    browser_enabled = True
    return code_languages, browser_enabled, api_functions


def validate_trajectory_metadata(
    trajectory: Any,
    metadata: DatasetMetadata,
    *,
    dataset_name: str | None = None,
    native_api_names: set[str] | None = None,
) -> None:
    native_api_names = native_api_names or set()
    dataset_label = dataset_name or "<unknown dataset>"
    tool_names = custom_tool_names(metadata)

    if trajectory.available_custom_tools is not None:
        available_custom_tools = set(trajectory.available_custom_tools)
        missing = sorted(available_custom_tools - tool_names)
        if missing:
            raise ValueError(
                f"available_custom_tools contains tools not found in metadata.json "
                f"for {dataset_label}: {missing}"
            )
    else:
        available_custom_tools = tool_names

    for item in trajectory.content:
        if isinstance(item, CodeAction):
            if item.language not in metadata.code_enabled:
                raise ValueError(
                    f"CodeAction language {item.language!r} is not enabled in "
                    f"metadata.json for {dataset_label}"
                )
        elif isinstance(item, WebObservation):
            if not metadata.browser_enabled:
                raise ValueError(
                    f"WebObservation appears but browser_enabled is false in "
                    f"metadata.json for {dataset_label}"
                )
        elif isinstance(item, ApiAction):
            if is_browser_api_action(
                item.function,
                item.kwargs,
                browser_context=metadata.browser_enabled,
            ):
                if not metadata.browser_enabled:
                    raise ValueError(
                        f"Browser ApiAction {item.function!r} appears but "
                        f"browser_enabled is false in metadata.json for {dataset_label}"
                    )
                continue
            if item.function in native_api_names:
                continue
            if item.function not in tool_names:
                raise ValueError(
                    f"ApiAction function {item.function!r} is not declared in "
                    f"metadata.json custom_tools for {dataset_label}"
                )
            if item.function not in available_custom_tools:
                raise ValueError(
                    f"ApiAction function {item.function!r} is missing from "
                    f"available_custom_tools for trajectory {trajectory.id!r}"
                )
