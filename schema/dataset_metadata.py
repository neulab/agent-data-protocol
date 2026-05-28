import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DATASET_ROOT = Path(__file__).parent.parent / "datasets"

BROWSER_ACTION_NAMES = {
    "back",
    "click",
    "fill",
    "go_back",
    "goto",
    "scroll",
    "type",
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
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class OpenAIToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["function"] = "function"
    function: OpenAIFunctionSpec


class DatasetMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    custom_tools: list[OpenAIToolSpec] = Field(default_factory=list)
    code_enabled: list[str] = Field(default_factory=list)
    browser_enabled: bool = False

    @model_validator(mode="after")
    def validate_unique_custom_tool_names(self):
        names = [tool.function.name for tool in self.custom_tools]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"custom_tools contains duplicate tool names: {duplicates}")
        self.code_enabled = sorted(set(self.code_enabled))
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


def custom_tool_map(metadata: DatasetMetadata) -> dict[str, OpenAIToolSpec]:
    return {tool.function.name: tool for tool in metadata.custom_tools}


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
    if function_name in {"back", "go_back"}:
        return False
    return bool(BROWSER_KWARG_NAMES & set(kwargs))
