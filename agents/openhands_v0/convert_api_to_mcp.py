import inspect
import textwrap
from typing import (
    Any,
    Optional,
    get_type_hints,
)

from pydantic import TypeAdapter

from schema.dataset_metadata import custom_tool_map, load_dataset_metadata


def json_type_from_py(py_t: Any) -> dict:
    """Generate JSON schema from Python type using Pydantic's TypeAdapter."""
    try:
        return TypeAdapter(py_t).json_schema()
    except Exception:
        # Fallback for unsupported types
        return {"type": "string"}


def split_docstring(ds: str):
    if not ds:
        return "", {}

    ds = textwrap.dedent(ds).strip()
    parts = ds.split("\n")
    summary_lines, arg_block_lines = [], []
    in_args = False
    for line in parts:
        if line.strip().startswith(("Args:", "Parameters:")):
            in_args = True
            continue
        (arg_block_lines if in_args else summary_lines).append(line)

    summary = "\n".join(summary_lines).strip()

    # Parse a lightweight Google-style args block
    arg_desc = {}
    current, buf = None, []
    for line in arg_block_lines:
        if not line.strip():
            if current and buf:
                arg_desc[current] = " ".join(s.strip() for s in buf).strip()
                buf = []
            continue
        if line.lstrip() == line and "(" in line and "):" in line:
            if current and buf:
                arg_desc[current] = " ".join(s.strip() for s in buf).strip()
                buf = []
            header, after = line.split("):", 1)
            name = header.split("(")[0].strip()
            current = name
            if after.strip():
                buf.append(after.strip())
        else:
            buf.append(line.strip())
    if current and buf:
        arg_desc[current] = " ".join(s.strip() for s in buf).strip()

    return summary, arg_desc


def tool_from_function(
    fn,
    *,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
) -> dict:
    sig = inspect.signature(fn)
    hints = get_type_hints(fn, globalns=fn.__globals__, include_extras=True)
    doc = inspect.getdoc(fn) or ""
    summary, arg_descs = split_docstring(doc)
    if description_override:
        summary = description_override
    if not summary:
        summary = f"Tool wrapping function '{fn.__name__}'."

    properties = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            # Skip *args/**kwargs
            continue

        py_t = hints.get(pname, str)
        schema = json_type_from_py(py_t)

        desc = arg_descs.get(pname, "").strip()
        if desc:
            schema["description"] = desc

        properties[pname] = schema

        # Required iff there is NO default
        if param.default is inspect._empty:
            required.append(pname)

    tool_name = name_override or fn.__name__

    schema_obj = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema_obj["required"] = required

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": summary,
            "parameters": schema_obj,
        },
    }


def get_api_tools(dataset) -> dict:
    metadata = load_dataset_metadata(dataset)
    return {
        name: tool.model_dump(exclude_none=True) for name, tool in custom_tool_map(metadata).items()
    }


def language_tool_placeholder(code: str):
    """ """
    pass


def get_language_tools(languages) -> dict:
    language_tools = {}
    for lan in languages:
        name_override = f"execute_{lan}"
        description_override = f"Execute {lan} code.\nEnsure your {lan} is valid {lan} code."
        language_tools[name_override] = tool_from_function(
            language_tool_placeholder,
            name_override=name_override,
            description_override=description_override,
        )
    return language_tools
