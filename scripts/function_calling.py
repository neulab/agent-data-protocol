"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

import json
import os
from browsergym.core.action.highlevel import HighLevelActionSet
import inspect
import importlib.util

dataset = os.getenv("MY_DATASET")
assert dataset, "Please set the environment variable MY_DATASET"

    
_BASH_DESCRIPTION = """Execute a bash command in the terminal.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command like `C-c` (Ctrl+C) to interrupt the process.
"""

CmdRunTool = {'name' : 'execute_bash',
        'description' : _BASH_DESCRIPTION,
        'parameters' : {
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command to execute. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.',
                },
            },
            'required': ['command'],
        },
}

api_file_path = os.path.expanduser(f"datasets/{dataset}/api.py")
_API_TOOL_DESCRIPTION = ""
if os.path.exists(api_file_path):
    spec = importlib.util.spec_from_file_location("api", api_file_path)
    api_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api_module)
    functions = inspect.getmembers(api_module, inspect.isfunction)
    for name, func in functions:
        docstring = '\n' + inspect.getdoc(func)
        sig = inspect.signature(func)
        docstring = f"{name}{sig}" + docstring.replace("\n", "\n    ") + '\n\n'   
        _API_TOOL_DESCRIPTION += docstring

_IPYTHON_DESCRIPTION = """Run a cell of Python code in an IPython environment.
* The assistant should define variables and import packages before using them.
* The variable defined in the IPython environment will not be available outside the IPython environment (e.g., in terminal).
"""

def getIPythonTool(is_web: bool):
    if not is_web and _API_TOOL_DESCRIPTION != '':
        _IPYTHON_TOOL_DESCRIPTION = """
        The following pre-defined functions are also available. 

        """ + _API_TOOL_DESCRIPTION
    else: _IPYTHON_TOOL_DESCRIPTION = ''
    IPythonTool = {
        "name": "execute_ipython_cell",
        "description": _IPYTHON_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute. Supports magic commands like %pip.\n" + _IPYTHON_TOOL_DESCRIPTION,
                },
            },
            "required": ["code"],
        }
    }
    return IPythonTool


_FILE_EDIT_DESCRIPTION = """Edit a file.
* The assistant can edit files by specifying the file path and providing a draft of the new file content.
* The draft content doesn't need to be exactly the same as the existing file; the assistant may skip unchanged lines using comments like `# unchanged` to indicate unchanged sections.
* IMPORTANT: For large files (e.g., > 300 lines), specify the range of lines to edit using `start` and `end` (1-indexed, inclusive). The range should be smaller than 300 lines.
* To append to a file, set both `start` and `end` to `-1`.
* If the file doesn't exist, a new file will be created with the provided content.

**Example 1: general edit for short files**
For example, given an existing file `/path/to/file.py` that looks like this:
(this is the end of the file)
1|class MyClass:
2|    def __init__(self):
3|        self.x = 1
4|        self.y = 2
5|        self.z = 3
6|
7|print(MyClass().z)
8|print(MyClass().x)
(this is the end of the file)

The assistant wants to edit the file to look like this:
(this is the end of the file)
1|class MyClass:
2|    def __init__(self):
3|        self.x = 1
4|        self.y = 2
5|
6|print(MyClass().y)
(this is the end of the file)

The assistant may produce an edit action like this:
path="/path/to/file.txt" start=1 end=-1
content=```
class MyClass:
    def __init__(self):
        # no changes before
        self.y = 2
        # self.z is removed

# MyClass().z is removed
print(MyClass().y)
```

**Example 2: append to file for short files**
For example, given an existing file `/path/to/file.py` that looks like this:
(this is the end of the file)
1|class MyClass:
2|    def __init__(self):
3|        self.x = 1
4|        self.y = 2
5|        self.z = 3
6|
7|print(MyClass().z)
8|print(MyClass().x)
(this is the end of the file)

To append the following lines to the file:
```python
print(MyClass().y)
```

The assistant may produce an edit action like this:
path="/path/to/file.txt" start=-1 end=-1
content=```
print(MyClass().y)
```

**Example 3: edit for long files**

Given an existing file `/path/to/file.py` that looks like this:
(1000 more lines above)
1001|class MyClass:
1002|    def __init__(self):
1003|        self.x = 1
1004|        self.y = 2
1005|        self.z = 3
1006|
1007|print(MyClass().z)
1008|print(MyClass().x)
(2000 more lines below)

The assistant wants to edit the file to look like this:

(1000 more lines above)
1001|class MyClass:
1002|    def __init__(self):
1003|        self.x = 1
1004|        self.y = 2
1005|
1006|print(MyClass().y)
(2000 more lines below)

The assistant may produce an edit action like this:
path="/path/to/file.txt" start=1001 end=1008
content=```
class MyClass:
    def __init__(self):
        # no changes before
        self.y = 2
        # self.z is removed

# MyClass().z is removed
print(MyClass().y)
```
"""

LLMBasedFileEditTool = {
    "name": "edit_file",
    "description": _FILE_EDIT_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The absolute path to the file to be edited.",
            },
            "content": {
                "type": "string",
                "description": "A draft of the new content for the file being edited. Note that the assistant may skip unchanged lines.",
            },
            "start": {
                "type": "integer",
                "description": "The starting line number for the edit (1-indexed, inclusive). Default is 1.",
            },
            "end": {
                "type": "integer",
                "description": "The ending line number for the edit (1-indexed, inclusive). Default is -1 (end of file).",
            },
        },
        "required": ["path", "content"],
    },
}


_STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
"""

StrReplaceEditorTool = {
    "name": "str_replace_editor",
    "description": _STR_REPLACE_EDITOR_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.",
                "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                "type": "string",
            },
            "path": {
                "description": "Absolute path to file or directory, e.g. `/workspace/file.py` or `/workspace`.",
                "type": "string",
            },
            "file_text": {
                "description": "Required parameter of `create` command, with the content of the file to be created.",
                "type": "string",
            },
            "old_str": {
                "description": "Required parameter of `str_replace` command containing the string in `path` to replace.",
                "type": "string",
            },
            "new_str": {
                "description": "Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.",
                "type": "string",
            },
            "insert_line": {
                "description": "Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.",
                "type": "integer",
            },
            "view_range": {
                "description": "Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.",
                "items": {"type": "integer"},
                "type": "array",
            },
        },
        "required": ["command", "path"],
    },
}


_WEB_DESCRIPTION = """Read (convert to markdown) content from a webpage. You should prefer using the `web_read` tool over the `browser` tool, but do use the `browser` tool if you need to interact with a webpage (e.g., click a button, fill out a form, etc.).

You may use the `web_read` tool to read content from a webpage, and even search the webpage content using a Google search query (e.g., url=`https://www.google.com/search?q=YOUR_QUERY`).
"""

WebReadTool = {
    "name": "web_read",
    "description": _WEB_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL of the webpage to read. You can also use a Google search query here (e.g., `https://www.google.com/search?q=YOUR_QUERY`).",
            },
        },
        "required": ["url"],
    },
}


# from browsergym/core/action/highlevel.py
_browser_action_space = HighLevelActionSet(
    subsets=['bid', 'nav'],
    strict=False,  # less strict on the parsing of the actions
    multiaction=True,  # enable to agent to take multiple actions at once
)


_BROWSER_DESCRIPTION = """Interact with the browser using Python code. Use it ONLY when you need to interact with a webpage.

See the description of "code" parameter for more details.

Multiple actions can be provided at once, but will be executed sequentially without any feedback from the page.
More than 2-3 actions usually leads to failure or unexpected behavior. Example:
fill('a12', 'example with "quotes"')
click('a51')
click('48', button='middle', modifiers=['Shift'])
"""

def getBrowserTool(is_web: bool):
    if is_web and _API_TOOL_DESCRIPTION != '':
        _BROWSER_TOOL_DESCRIPTION = """
        The following functions are available. Nothing else is supported.

        """ + _API_TOOL_DESCRIPTION
    else: _BROWSER_TOOL_DESCRIPTION = ''
    BrowserTool = {
        "name": "browser",
        "description": _BROWSER_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "The Python code that interacts with the browser.\n" + _BROWSER_TOOL_DESCRIPTION
                    ),
                },
            },
            "required": ["code"],
        },
    }
    return BrowserTool


_FINISH_DESCRIPTION = """Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task."""

FinishTool = {
        'name':'finish',
        'description':_FINISH_DESCRIPTION,
}


def get_tools(
    codeact_enable_browsing: bool = False,
    codeact_enable_llm_editor: bool = False,
    codeact_enable_jupyter: bool = False,
    is_web: bool = False,
) -> list:
    tools = [CmdRunTool, FinishTool, StrReplaceEditorTool]
    if codeact_enable_browsing:
        tools.append(WebReadTool)
        tools.append(getBrowserTool(is_web))
    if codeact_enable_llm_editor:
        tools.append(LLMBasedFileEditTool)
    if codeact_enable_jupyter:
        tools.append(getIPythonTool(is_web))
    return tools
