from schema.dataset_metadata import load_dataset_metadata, openai_tool_signature

sweagent_default_tools = {
    "bash": {"required": ["command"], "optional": []},
    "submit": {"required": [], "optional": []},
    "str_replace_editor": {
        "required": ["command", "path"],
        "optional": ["file_text", "old_str", "new_str", "insert_line", "view_range"],
    },
}


def check_exclude_sweagent_default_tools(name, sig, required, optional):
    if not all(
        api in sweagent_default_tools[name]["required"] + sweagent_default_tools[name]["optional"]
        for api in required
    ):
        # print(f"mismatch required arguments: {name}, {sig}")
        return False
    if not all(api in sweagent_default_tools[name]["optional"] for api in optional):
        # print(f"mismatch optional arguments: {name}, {sig}")
        return False
    if not all(api in required for api in sweagent_default_tools[name]["required"]):
        # print(f"mismatch required arguments: {name}, {sig}")
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
        # print(f"{name} is included")
        return False
    if not all(api in exclude_api_optional for api in optional):
        # print(f"{name} is included")
        return False
    if not all(api in required for api in exclude_api_required):
        # print(f"{name} is included")
        return False
    return True


def get_api_tool_description(dataset, exclude_apis=None, env="bash", include_custom_tools=None):
    if exclude_apis is None:
        exclude_apis = {}
    if include_custom_tools is not None:
        if not isinstance(include_custom_tools, list) or not all(
            isinstance(api_name, str) for api_name in include_custom_tools
        ):
            raise ValueError("available_custom_tools must be a list of custom tool names")
        include_api_names = set(include_custom_tools)
    else:
        include_api_names = None

    metadata = load_dataset_metadata(dataset)
    API_TOOL_DESCRIPTION = ""
    tools = metadata.custom_tools
    if tools:
        if include_api_names is not None:
            api_names = {tool.function.name for tool in tools}
            missing_api_names = sorted(include_api_names - api_names)
            if missing_api_names:
                raise ValueError(
                    f"available_custom_tools contains tools not found in metadata.json: "
                    f"{missing_api_names}"
                )
        sigs = {}
        for tool in tools:
            name = tool.function.name
            if include_api_names is not None and name not in include_api_names:
                continue
            required, optional, signature = openai_tool_signature(tool)
            if name in sweagent_default_tools and check_exclude_sweagent_default_tools(
                name, signature, required, optional
            ):
                # print(f"excluded {name}")
                continue
            if name in exclude_apis and check_exclude_tools(name, required, optional, exclude_apis):
                # print(f"excluded {name}")
                continue
            description = "\n" + (tool.function.description or "")
            docstring = f"{signature}" + description.replace("\n", "\n    ") + "\n\n"
            API_TOOL_DESCRIPTION += docstring
            sigs[name] = {"required": required, "optional": optional}
        if not API_TOOL_DESCRIPTION:
            return "", {}
        if exclude_apis:
            also = "also "
        else:
            also = ""
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
        API_TOOL_DESCRIPTION = prefixes[0] + "\n\n" + API_TOOL_DESCRIPTION
        API_TOOL_DESCRIPTION = API_TOOL_DESCRIPTION.replace("xpath", "bid").replace(
            "element_id", "bid"
        )
        return API_TOOL_DESCRIPTION, sigs
    else:
        return "", {}
