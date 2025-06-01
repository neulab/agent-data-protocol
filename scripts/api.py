import importlib.util
import inspect
import os


def get_api_tool_description(dataset, env=None):
    api_file_path = os.path.expanduser(f"datasets/{dataset}/api.py")
    API_TOOL_DESCRIPTION = f"The following pre-defined functions are available in {env}. \n\n"
    if os.path.exists(api_file_path):
        spec = importlib.util.spec_from_file_location("api", api_file_path)
        api_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api_module)
        functions = inspect.getmembers(api_module, inspect.isfunction)
        for name, func in functions:
            docstring = "\n" + inspect.getdoc(func)
            sig = inspect.signature(func)
            docstring = f"{name}{sig}" + docstring.replace("\n", "\n    ") + "\n\n"
            API_TOOL_DESCRIPTION += docstring
        return API_TOOL_DESCRIPTION
    else:
        return ""
