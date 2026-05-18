import os

from agents.openhands.system_prompt.tools import (
    BrowserTool,
    FinishTool,
    IPythonTool,
    LLMBasedFileEditTool,
    ThinkTool,
    WebReadTool,
    create_cmd_run_tool,
    create_str_replace_editor_tool,
)

_script_dir = os.path.dirname(os.path.realpath(__file__))
prompt_file = os.path.join(_script_dir, "system_prefix.txt")


SYSTEM_PROMPT_SUFFIX_TEMPLATE = """
You have access to the following functions:

{description}

If you choose to call a function ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after.
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>
"""


def get_tools(
    codeact_enable_browsing: bool = True,
    codeact_enable_llm_editor: bool = False,
    codeact_enable_jupyter: bool = True,
) -> list:
    use_simplified_tool_desc = False

    tools = [
        create_cmd_run_tool(use_simplified_description=use_simplified_tool_desc),
        ThinkTool,
        FinishTool,
    ]
    if codeact_enable_browsing:
        tools.append(WebReadTool)
        tools.append(BrowserTool)
    if codeact_enable_jupyter:
        tools.append(IPythonTool)
    if codeact_enable_llm_editor:
        tools.append(LLMBasedFileEditTool)
    else:
        tools.append(
            create_str_replace_editor_tool(use_simplified_description=use_simplified_tool_desc)
        )
    return tools


def convert_tools_to_description(tools: list[dict]) -> str:
    ret = ""
    for i, tool in enumerate(tools):
        try:
            assert tool["type"] == "function"
        except Exception:
            raise ValueError(f"tool ({type(tool)}): {tool}")
        fn = tool["function"]
        if i > 0:
            ret += "\n"
        ret += f"---- BEGIN FUNCTION #{i + 1}: {fn['name']} ----\n"
        ret += f"Description: {fn['description']}\n"

        if "parameters" in fn:
            ret += "Parameters:\n"
            properties = fn["parameters"].get("properties", {})
            required_params = set(fn["parameters"].get("required", []))

            for j, (param_name, param_info) in enumerate(properties.items()):
                # Indicate required/optional in parentheses with type
                is_required = param_name in required_params
                param_status = "required" if is_required else "optional"
                param_type = param_info.get("type", "string")

                # Get parameter description
                desc = param_info.get("description", "")

                # Handle enum values if present
                if "enum" in param_info:
                    enum_values = ", ".join(f"`{v}`" for v in param_info["enum"])
                    desc += f"\nAllowed values: [{enum_values}]"

                if desc:
                    ret += f"  ({j + 1}) {param_name} ({param_type}, {param_status}): {desc}\n"
                else:
                    ret += f"  ({j + 1}) {param_name} ({param_type}, {param_status})\n"
        else:
            ret += "No parameters are required for this function.\n"

        ret += f"---- END FUNCTION #{i + 1} ----\n"
    return ret


def get_system_message(
    codeact_enable_browsing: bool = True,
    codeact_enable_llm_editor: bool = False,
    codeact_enable_jupyter: bool = True,
    additional_tools=[],
):
    with open(prompt_file, "r") as f:
        system_prompt = f.read()
    tools = get_tools(
        codeact_enable_browsing=codeact_enable_browsing,
        codeact_enable_llm_editor=codeact_enable_llm_editor,
        codeact_enable_jupyter=codeact_enable_jupyter,
    )
    tools += additional_tools
    formatted_tools = convert_tools_to_description(tools)
    system_prompt_suffix = SYSTEM_PROMPT_SUFFIX_TEMPLATE.format(description=formatted_tools)
    system_prompt += system_prompt_suffix
    return system_prompt
