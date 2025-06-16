import importlib.util
import inspect
import os
import random

openhands_default_tools = {
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


def check_exclude_openhands_default_tools(name, sig, required, optional):
    if not all(
        api in openhands_default_tools[name]["required"] + openhands_default_tools[name]["optional"]
        for api in required
    ):
        print(f"mismatch required arguments: {name}, {sig}")
        return False
    if not all(api in openhands_default_tools[name]["optional"] for api in optional):
        print(f"mismatch optional arguments: {name}, {sig}")
        return False
    if not all(api in required for api in openhands_default_tools[name]["required"]):
        print(f"mismatch required arguments: {name}, {sig}")
        return False
    return True


def check_exclude_tools(name: str, required: list, optional: list, exclude_apis: dict):
    exclude_api_required = exclude_apis[name]["required"]
    exclude_api_optional = exclude_apis[name]["optional"]
    if ("id" in required or "id" in optional) and "bid" in exclude_api_required:
        required.remove("id")
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


def get_api_tool_description(dataset, exclude_apis={}, env="execute_ipython_cell"):
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
                    required.append(arg_name)
                else:
                    optional.append(arg_name)
            if name in openhands_default_tools and check_exclude_openhands_default_tools(
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
        return API_TOOL_DESCRIPTION, sigs
    else:
        return "", {}


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
