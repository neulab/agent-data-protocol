import importlib.util
import inspect
import textwrap
import json
import os
from typing import (
    get_type_hints,
    get_origin,
    get_args,
    Optional,
    Union,
    Literal,
    Any,
)


def json_type_from_py(py_t: Any) -> dict:
    origin = get_origin(py_t)
    args = get_args(py_t)

    # Literal[...] -> enum
    if origin is Literal and args:
        base_types = set(type(a) for a in args if a is not None)
        if base_types == {str}:
            return {"type": "string", "enum": list(args)}
        if base_types == {int}:
            return {"type": "integer", "enum": list(args)}
        if base_types == {float}:
            return {"type": "number", "enum": list(args)}
        if base_types == {bool}:
            return {"type": "boolean", "enum": list(args)}
        # Mixed/other: just enum without type
        return {"enum": list(args)}

    # Optional[T] / Union[T, None]
    if origin is Union and type(None) in args:
        inner = [a for a in args if a is not type(None)]
        return json_type_from_py(inner[0]) if inner else {"type": "null"}

    # General Union[...] -> anyOf
    if origin is Union:
        return {"anyOf": [json_type_from_py(a) for a in args]}

    # Containers
    if origin in (list, tuple):
        item_t = args[0] if args else str
        return {"type": "array", "items": json_type_from_py(item_t)}
    if origin is dict:
        val_t = args[1] if len(args) == 2 else Any
        schema = {"type": "object"}
        schema["additionalProperties"] = json_type_from_py(val_t)
        return schema

    # Primitives
    if py_t is str:
        return {"type": "string"}
    if py_t is int:
        return {"type": "integer"}
    if py_t is float:
        return {"type": "number"}
    if py_t is bool:
        return {"type": "boolean"}

    # Fallback
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
    api_file_path = os.path.expanduser(f"datasets/{dataset}/api.py")
    if os.path.exists(api_file_path):
        api_tools = {}
        spec = importlib.util.spec_from_file_location("api", api_file_path)
        api_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api_module)
        functions = inspect.getmembers(api_module, inspect.isfunction)
        for name, func in functions:
            api_tools[name] = tool_from_function(func)
        return api_tools
    else: 
        return {}


def language_tool_placeholder(code: str):
    """
    """
    pass


def get_language_tools(languages) -> dict:
    language_tools = {}
    for lan in languages:
        name_override = f"execute_{lan}"
        description_override = (
            f"Execute {lan} code.\n"
            f"Ensure your {lan} is valid {lan} code."
        )
        language_tools[name_override] = tool_from_function(language_tool_placeholder, name_override=name_override, description_override=description_override)
    return language_tools
