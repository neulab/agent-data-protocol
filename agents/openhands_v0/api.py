from schema.dataset_metadata import custom_tool_map, load_dataset_metadata

openhands_v0_default_tools = {
    "execute_bash": {"required": ["command"], "optional": ["is_input"]},
    "think": {"required": ["thought"], "optional": []},
    "finish": {"required": ["message", "task_completed"], "optional": []},
    "web_read": {"required": ["url"], "optional": []},
    "browser": {"required": ["code"], "optional": []},
    "execute_ipython_cell": {"code": ["command"], "optional": []},
    "str_replace_editor": {
        "required": ["command", "path"],
        "optional": ["file_text", "old_str", "new_str", "insert_line", "view_range"],
    },
    "edit_file": {"required": ["path", "content"], "optional": ["start", "end"]},
}

browser_default_apis = {
    "goto": {"required": ["url"], "optional": []},
    "go_back": {"required": [], "optional": []},
    "go_forward": {"required": [], "optional": []},
    "noop": {"required": [], "optional": ["wait_ms"]},
    "scroll": {"required": ["delta_x", "delta_y"], "optional": []},
    "fill": {"required": ["bid", "value"], "optional": []},
    "select_option": {"required": ["bid", "options"], "optional": []},
    "click": {"required": ["bid"], "optional": ["button", "modifiers"]},
    "dblclick": {"required": ["bid"], "optional": ["button", "modifiers"]},
    "hover": {"required": ["bid"], "optional": []},
    "press": {"required": ["bid", "key_comb"], "optional": []},
    "focus": {"required": ["bid"], "optional": []},
    "clear": {"required": ["bid"], "optional": []},
    "drag_and_drop": {"required": ["from_bid", "to_bid"], "optional": []},
    "upload_file": {"required": ["bid", "file"], "optional": []},
}


def check_exclude_openhands_v0_default_tools(name, sig, required, optional):
    if not all(
        api
        in openhands_v0_default_tools[name]["required"]
        + openhands_v0_default_tools[name]["optional"]
        for api in required
    ):
        # print(f"mismatch required arguments: {name}, {sig}", file=sys.stderr)
        return False
    if not all(api in openhands_v0_default_tools[name]["optional"] for api in optional):
        # print(f"mismatch optional arguments: {name}, {sig}", file=sys.stderr)
        return False
    if not all(api in required for api in openhands_v0_default_tools[name]["required"]):
        # print(f"mismatch required arguments: {name}, {sig}", file=sys.stderr)
        return False
    return True


def check_exclude_tools(name: str, required: list, optional: list, exclude_apis: dict):
    exclude_api_required = exclude_apis[name]["required"]
    exclude_api_optional = exclude_apis[name]["optional"]
    if ("id" in required or "id" in optional) and "bid" in exclude_api_required:
        required.remove("id")
        required.append("bid")
    elif ("xpath" in required or "xpath" in optional) and "bid" in exclude_api_required:
        required.remove("xpath")
        required.append("bid")
    elif ("element_id" in required or "element_id" in optional) and "bid" in exclude_api_required:
        required.remove("element_id")
        required.append("bid")
    if not all(api in exclude_api_required + exclude_api_optional for api in required):
        # print(f"{name} is included", file=sys.stderr)
        return False
    if not all(api in exclude_api_optional for api in optional):
        # print(f"{name} is included", file=sys.stderr)
        return False
    if not all(api in required for api in exclude_api_required):
        # print(f"{name} is included", file=sys.stderr)
        return False
    return True


def _schema_signature(tool) -> tuple[str, list[str], list[str]]:
    parameters = tool.function.parameters or {}
    properties = parameters.get("properties", {}) or {}
    required = list(parameters.get("required", []) or [])
    optional = [name for name in properties if name not in required]
    args = [*required, *(f"{name}=None" for name in optional)]
    return f"({', '.join(args)})", required, optional


def _tool_docstring(tool) -> str:
    description = tool.function.description or ""
    return "\n" + description


def get_api_tool_description(
    dataset, exclude_apis=None, env="execute_ipython_cell", include_apis=None
):
    if exclude_apis is None:
        exclude_apis = {}
    if include_apis is not None:
        if not isinstance(include_apis, list) or not all(
            isinstance(api_name, str) for api_name in include_apis
        ):
            raise ValueError("available_apis must be a list of API function names")
        include_api_names = set(include_apis)
    else:
        include_api_names = None

    metadata = load_dataset_metadata(dataset)
    tools = custom_tool_map(metadata)
    if include_api_names is not None:
        missing_api_names = sorted(include_api_names - set(tools))
        if missing_api_names:
            raise ValueError(
                f"available_apis contains functions not found in metadata.json for "
                f"{dataset}: {missing_api_names}"
            )

    api_tool_description = ""
    sigs = {}
    for name, tool in sorted(tools.items()):
        if include_api_names is not None and name not in include_api_names:
            continue
        sig, required, optional = _schema_signature(tool)
        if name in openhands_v0_default_tools and check_exclude_openhands_v0_default_tools(
            name, sig, required, optional
        ):
            continue
        if name in exclude_apis and check_exclude_tools(name, required, optional, exclude_apis):
            continue
        docstring = f"{name}{sig}" + _tool_docstring(tool).replace("\n", "\n    ") + "\n\n"
        api_tool_description += docstring
        sigs[name] = {"required": required, "optional": optional}

    if not api_tool_description:
        return "", {}
    also = "also " if exclude_apis else ""
    prefixes = [
        f"The following pre-defined functions are {also}available in {env}. ",
        f"The environment {env} {also}provides the following pre-defined functions: ",
        f"In {env}, you can {also}use the following pre-defined functions: ",
        f"Available functions in {env}: ",
        f"The following functions are {also}defined and ready for use in {env}: ",
        f"Note that {env} {also}supports the following pre-defined functions: ",
        f"Below is a list of functions you can {also}use in the {env} environment. ",
        f"The toolkit for {env} {also}contains the following functions. ",
    ]
    api_tool_description = prefixes[0] + "\n\n" + api_tool_description
    api_tool_description = api_tool_description.replace("xpath", "bid").replace("element_id", "bid")
    return api_tool_description, sigs


def get_language_descriptions(languages):
    language_description = ""
    for lan in languages:
        language_description += (
            f"In the execute_ipython_cell code environment, you can execute {lan} code by wrapping it in the following format: "
            f"{lan}('YOUR {lan.upper()} CODE')\n"
            f"The {lan} code must be provided as a quoted string inside the {lan}(...) function. "
            f"Ensure 'YOUR {lan.upper()} CODE' is valid {lan} code.\n\n"
        )
    return language_description.strip()
