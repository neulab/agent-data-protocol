import importlib.util
import inspect
import os
import random

sweagent_default_tools = {
    "bash": {"required": ["command"], "optional": []},
    "submit": {"required": [], "optional": []},
    "str_replace_editor": {
        "required": ["command", "path"],
        "optional": ["file_text", "old_str", "new_str", "insert_line", "view_range"],
    },
}


def check_exclude_openhands_default_tools(name, sig, required, optional):
    if not all(
        api in sweagent_default_tools[name]["required"] + sweagent_default_tools[name]["optional"]
        for api in required
    ):
        print(f"mismatch required arguments: {name}, {sig}")
        return False
    if not all(api in sweagent_default_tools[name]["optional"] for api in optional):
        print(f"mismatch optional arguments: {name}, {sig}")
        return False
    if not all(api in required for api in sweagent_default_tools[name]["required"]):
        print(f"mismatch required arguments: {name}, {sig}")
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
        print(f"{name} is included")
        return False
    if not all(api in exclude_api_optional for api in optional):
        print(f"{name} is included")
        return False
    if not all(api in required for api in exclude_api_required):
        print(f"{name} is included")
        return False
    return True


def get_api_tool_description(dataset, exclude_apis={}, env="bash"):
    api_file_path = os.path.expanduser(f"datasets/{dataset}/api.py")
    API_TOOL_DESCRIPTION = ""
    if os.path.exists(api_file_path):
        spec = importlib.util.spec_from_file_location("api", api_file_path)
        api_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api_module)
        functions = inspect.getmembers(api_module, inspect.isfunction)
        sigs = {}
        for name, func in functions:
            docstring = "\n" + inspect.getdoc(func)
            sig = inspect.signature(func)
            required = []
            optional = []
            for arg_name, param in sig.parameters.items():
                if param.default is inspect.Parameter.empty:
                    if arg_name not in required:
                        required.append(arg_name)
                else:
                    optional.append(arg_name)
            if name in sweagent_default_tools and check_exclude_openhands_default_tools(
                name, sig, required, optional
            ):
                print(f"excluded {name}")
                continue
            if name in exclude_apis and check_exclude_tools(name, required, optional, exclude_apis):
                print(f"excluded {name}")
                continue
            docstring = f"{name}{sig}" + docstring.replace("\n", "\n    ") + "\n\n"
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
        API_TOOL_DESCRIPTION = random.choice(prefixes) + "\n\n" + API_TOOL_DESCRIPTION
        API_TOOL_DESCRIPTION = API_TOOL_DESCRIPTION.replace("xpath", "bid").replace(
            "element_id", "bid"
        )
        return API_TOOL_DESCRIPTION, sigs
    else:
        return "", {}
