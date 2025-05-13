import os
from scripts.system_prompt.get_tools import get_tools, convert_tools_to_description

_script_dir = os.path.dirname(os.path.realpath(__file__))
prompt_file = os.path.join(_script_dir, 'system_prefix.txt')

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


def get_system_message(
    codeact_enable_browsing: bool = True,
    codeact_enable_llm_editor: bool = False,
    codeact_enable_jupyter: bool = True,
):
    with open(prompt_file, 'r') as f:
        system_prompt = f.read()
    tools = get_tools(codeact_enable_browsing=codeact_enable_browsing,
                      codeact_enable_llm_editor=codeact_enable_llm_editor,
                      codeact_enable_jupyter=codeact_enable_jupyter)
    formatted_tools = convert_tools_to_description(tools)
    system_prompt_suffix = SYSTEM_PROMPT_SUFFIX_TEMPLATE.format(
        description=formatted_tools
    )
    system_prompt += system_prompt_suffix
    return system_prompt
