from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence, Self

from pydantic import SecretStr

from openhands.sdk import ImageContent, LLM, Message, Observation, TextContent, ToolDefinition
from openhands.sdk.llm import MessageToolCall
from openhands.sdk.tool import Action, ToolAnnotations, ToolExecutor
from openhands.sdk.tool.builtins.finish import FinishTool
from openhands.sdk.tool.builtins.think import ThinkTool
from openhands.tools.browser_use.definition import (
    BROWSER_CLICK_DESCRIPTION,
    BROWSER_GET_CONTENT_DESCRIPTION,
    BROWSER_GET_STATE_DESCRIPTION,
    BROWSER_GO_BACK_DESCRIPTION,
    BROWSER_NAVIGATE_DESCRIPTION,
    BROWSER_SCROLL_DESCRIPTION,
    BROWSER_TYPE_DESCRIPTION,
    BrowserClickAction,
    BrowserClickTool,
    BrowserGetContentAction,
    BrowserGetContentTool,
    BrowserGetStateAction,
    BrowserGetStateTool,
    BrowserGoBackAction,
    BrowserGoBackTool,
    BrowserNavigateAction,
    BrowserNavigateTool,
    BrowserObservation,
    BrowserScrollAction,
    BrowserScrollTool,
    BrowserTypeAction,
    BrowserTypeTool,
)
from openhands.tools.file_editor.definition import (
    TOOL_DESCRIPTION as FILE_EDITOR_DESCRIPTION,
    FileEditorAction,
    FileEditorObservation,
    FileEditorTool,
)
from openhands.tools.task_tracker.definition import (
    TASK_TRACKER_DESCRIPTION,
    TaskTrackerAction,
    TaskTrackerObservation,
    TaskTrackerTool,
)
from openhands.tools.terminal.definition import TerminalAction, TerminalObservation, TerminalTool
from openhands.tools.terminal.descriptions import UNIX_TOOL_DESCRIPTION


DATASET_NAME = 'swe-play-trajectories'
MODEL = os.getenv("LLM_MODEL", "openhands/minimax-m2.7")
RECORD_ID = 'swe-play-0'
ASSISTANT_INDEX = 2
PREFIX_MESSAGES_JSON = '[{"content": [{"text": "You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.\\n\\n<ROLE>\\n* Your primary role is to assist users by executing commands, modifying code, and solving technical problems effectively. You should be thorough, methodical, and prioritize quality over speed.\\n* If the user asks a question, like \\"why is X happening\\", don\'t try to fix the problem. Just give an answer to the question.\\n</ROLE>\\n\\n<MEMORY>\\n* Use `AGENTS.md` under the repository root as your persistent memory for repository-specific knowledge and context.\\n* Add important insights, patterns, and learnings to this file to improve future task performance.\\n* This repository skill is automatically loaded for every conversation and helps maintain context across sessions.\\n</MEMORY>\\n\\n<EFFICIENCY>\\n* Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action, e.g. combine multiple bash commands into one, using sed and grep to edit/view multiple files at once.\\n* When exploring the codebase, use efficient tools like find, grep, and git commands with appropriate filters to minimize unnecessary operations.\\n</EFFICIENCY>\\n\\n<FILE_SYSTEM_GUIDELINES>\\n* When a user provides a file path, do NOT assume it\'s relative to the current working directory. First explore the file system to locate the file before working on it.\\n* If asked to edit a file, edit the file directly, rather than creating a new file with a different filename.\\n* For global search-and-replace operations, consider using `sed` instead of opening file editors multiple times.\\n* NEVER create multiple versions of the same file with different suffixes. Modify the original file directly when making changes.\\n</FILE_SYSTEM_GUIDELINES>\\n\\n<CODE_QUALITY>\\n* Write clean, efficient code with minimal comments.\\n* Before implementing changes, first thoroughly understand the codebase through exploration.\\n* Place imports at the top of the file unless doing so would cause circular imports or other real issues.\\n</CODE_QUALITY>\\n\\n<VERSION_CONTROL>\\n* Exercise caution with git operations. Do NOT make potentially dangerous changes unless explicitly asked to do so.\\n* When committing changes, use `git status` to see all modified files, and stage all files necessary for the commit.\\n* Do NOT commit files that typically should not go into version control unless explicitly instructed by the user.\\n</VERSION_CONTROL>\\n\\n<PROBLEM_SOLVING_WORKFLOW>\\n1. EXPLORATION: Thoroughly explore relevant files and understand the context before proposing solutions.\\n2. ANALYSIS: Consider multiple approaches and select the most promising one.\\n3. TESTING: For bug fixes, create tests to verify issues before implementing fixes when appropriate.\\n4. IMPLEMENTATION: Make focused, minimal changes to address the problem.\\n5. VERIFICATION: Test your implementation thoroughly, including edge cases.\\n</PROBLEM_SOLVING_WORKFLOW>\\n\\n<SECURITY_RISK_ASSESSMENT>\\nWhen using tools that support the security_risk parameter, assess the safety risk of your actions:\\n* LOW: Safe, read-only actions.\\n* MEDIUM: Project-scoped edits or execution.\\n* HIGH: System-level or untrusted operations, privilege changes, or sensitive-data movement.\\n</SECURITY_RISK_ASSESSMENT>\\n\\n<EXTERNAL_SERVICES>\\n* When interacting with external services like GitHub, GitLab, or Bitbucket, use their respective APIs instead of browser-based interactions whenever possible.\\n* Only resort to browser-based interactions with these services if specifically requested by the user or if the required operation cannot be performed via API.\\n</EXTERNAL_SERVICES>\\n\\n<ENVIRONMENT_SETUP>\\n* When user asks you to run an application, do not stop if the application is not installed. Instead, install the application and run the command again when possible.\\n* If you encounter missing dependencies, first look around in the repository for existing dependency files and use them to install dependencies.\\n</ENVIRONMENT_SETUP>\\n", "type": "text"}], "role": "system"}, {"content": [{"text": "Here\'s a running example of how to perform a task with the provided tools.\\n\\n--------------------- START OF EXAMPLE ---------------------\\n\\nUSER: Create a list of numbers from 1 to 10, and display them in a web page at port 5000.\\n\\n\\nASSISTANT: Sure! Let me first check the current directory:\\n<function=execute_bash>\\n<parameter=command>\\npwd && ls\\n</parameter>\\n</function>\\n\\nUSER: EXECUTION RESULT of [execute_bash]:\\n/workspace\\nopenhands@runtime:~/workspace$\\n\\nASSISTANT:\\nThere is no `app.py` file in the current directory. Let me create a Python file `app.py`:\\n<function=str_replace_editor>\\n<parameter=command>create</parameter>\\n<parameter=path>/workspace/app.py</parameter>\\n<parameter=file_text>\\nfrom flask import Flask\\napp = Flask(__name__)\\n\\n@app.route(\'/\')\\ndef index():\\n    numbers = list(range(1, 11))\\n    return str(numbers)\\n\\nif __name__ == \'__main__\':\\n    app.run(port=5000)\\n</parameter>\\n</function>\\n\\nUSER: EXECUTION RESULT of [str_replace_editor]:\\nFile created successfully at: /workspace/app.py\\n\\nASSISTANT:\\nLet me run the Python file for you:\\n<function=execute_bash>\\n<parameter=command>\\npython3 app.py > server.log 2>&1 &\\n</parameter>\\n</function>\\n\\nUSER: EXECUTION RESULT of [execute_bash]:\\n[1] 121\\n[1]+  Exit 1                  python3 app.py > server.log 2>&1\\n\\nASSISTANT:\\nLooks like the server was running with PID 121 then crashed. Let me check the server log:\\n<function=execute_bash>\\n<parameter=command>\\ncat server.log\\n</parameter>\\n</function>\\n\\nUSER: EXECUTION RESULT of [execute_bash]:\\nTraceback (most recent call last):\\n  File \\"/workspace/app.py\\", line 2, in <module>\\n    from flask import Flask\\nModuleNotFoundError: No module named \'flask\'\\n\\nASSISTANT:\\nLooks like the server crashed because the `flask` module is not installed. Let me install the `flask` module for you:\\n<function=execute_bash>\\n<parameter=command>\\npip3 install flask\\n</parameter>\\n</function>\\n\\nUSER: EXECUTION RESULT of [execute_bash]:\\nDefaulting to user installation because normal site-packages is not writeable\\nCollecting flask\\n  Using cached flask-3.0.3-py3-none-any.whl (101 kB)\\nCollecting blinker>=1.6.2\\n  Using cached blinker-1.7.0-py3-none-any.whl (13 kB)\\nCollecting Werkzeug>=3.0.0\\n  Using cached werkzeug-3.0.2-py3-none-any.whl (226 kB)\\nCollecting click>=8.1.3\\n  Using cached click-8.1.7-py3-none-any.whl (97 kB)\\nCollecting itsdangerous>=2.1.2\\n  Using cached itsdangerous-2.2.0-py3-none-any.whl (16 kB)\\nRequirement already satisfied: Jinja2>=3.1.2 in /home/openhands/.local/lib/python3.10/site-packages (from flask) (3.1.3)\\nRequirement already satisfied: MarkupSafe>=2.0 in /home/openhands/.local/lib/python3.10/site-packages (from Jinja2>=3.1.2->flask) (2.1.5)\\nInstalling collected packages: Werkzeug, itsdangerous, click, blinker, flask\\nSuccessfully installed Werkzeug-3.0.2 blinker-1.7.0 click-8.1.7 flask-3.0.3 itsdangerous-2.2.0\\n\\nASSISTANT:\\nThe `flask` module is installed successfully. Let me run the Python file again:\\n<function=execute_bash>\\n<parameter=command>\\npython3 app.py > server.log 2>&1 &\\n</parameter>\\n</function>\\n\\nUSER: EXECUTION RESULT of [execute_bash]:\\n[1] 124\\n\\nASSISTANT:\\nLet me check the server log again:\\n<function=execute_bash>\\n<parameter=command>\\ncat server.log\\n</parameter>\\n</function>\\n\\nUSER: EXECUTION RESULT of [execute_bash]:\\n* Serving Flask app \'app\'\\n * Debug mode: off\\nWARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\\n * Running on http://127.0.0.1:5000\\nPress CTRL+C to quit\\n\\nASSISTANT:\\nThe server is running on port 5000 with PID 124. You can access the list of numbers by visiting http://127.0.0.1:5000. If you have any further questions, feel free to ask!\\n\\nASSISTANT:\\nLet me check how the page looks in the browser:\\n<function=browser>\\n<parameter=code>\\ngoto(\'http://127.0.0.1:5000\')\\nnoop(1000)  # Wait for page to load\\n</parameter>\\n</function>\\n\\nUSER: EXECUTION RESULT of [browser]:\\n[Browser shows the numbers in a table format]\\n\\nUSER: Now kill the server, make it display the numbers in a table format.\\n\\nASSISTANT:\\nSure! Let me stop the server first:\\n<function=execute_bash>\\n<parameter=command>\\nkill 124\\n</parameter>\\n</function>\\n\\nUSER: EXECUTION RESULT of [execute_bash]:\\n[1]+  Terminated              python3 app.py > server.log 2>&1\\n\\nASSISTANT:\\nNow let me display the numbers in a table format:\\n<function=str_replace_editor>\\n<parameter=command>str_replace</parameter>\\n<parameter=path>/workspace/app.py</parameter>\\n<parameter=old_str>return str(numbers)</parameter>\\n<parameter=new_str>return \'<table>\' + \'\'.join([f\'<tr><td>{i}</td></tr>\' for i in numbers]) + \'</table>\'</parameter>\\n</function>\\n\\nUSER: EXECUTION RESULT of [str_replace_editor]:\\nThe file /workspace/app.py has been edited. Here\'s the result of running `cat -n` on a snippet of /workspace/app.py:\\n     3\\n     4  @app.route(\'/\')\\n     5  def index():\\n     6      numbers = list(range(1, 11))\\n     7      return \'<table>\' + \'\'.join([f\'<tr><td>{i}</td></tr>\' for i in numbers]) + \'</table>\'\\n     8\\n     9  if __name__ == \'__main__\':\\n    10      app.run(port=5000)\\nReview the changes and make sure they are as expected. Edit the file again if necessary.\\n\\n\\nASSISTANT:\\nRunning the updated file:\\n<function=execute_bash>\\n<parameter=command>\\npython3 app.py > server.log 2>&1 &\\n</parameter>\\n</function>\\n\\nUSER: EXECUTION RESULT of [execute_bash]:\\n[1] 126\\n\\nASSISTANT:\\nThe server is running on port 5000 with PID 126. You can access the list of numbers in a table format by visiting http://127.0.0.1:5000.\\n\\nASSISTANT:\\nThe server is running on port 5000 with PID 126. You can access the list of numbers in a table format by visiting http://127.0.0.1:5000. Let me know if you have any further requests!\\n<function=finish>\\n<parameter=message>The task has been completed. The web server is running and displaying numbers 1-10 in a table format at http://127.0.0.1:5000.</parameter>\\n<parameter=task_completed>true</parameter>\\n</function>\\n\\n--------------------- END OF EXAMPLE ---------------------\\n\\nDo NOT assume the environment is the same as in the example above.\\n\\n--------------------- NEW TASK DESCRIPTION ---------------------\\nYou are now given a project directory and required to finish the project task by task.\\nYou have finished all tasks before 1.2.1. Now it\'s your task to finish Task 1.2.1.\\n\\n<task_description>\\nCreate a CPU class that combines registers, memory, and program counter into a unified state container. Include methods to reset the CPU state, display current state, and provide access to individual components. This class will serve as the main interface for CPU operations.\\n</task_description>\\n\\nYou should finish the task based on the given <task_description>. Also, here are some constraints that you should follow:\\nCannot use any existing VM or CPU simulation frameworks. The assembler for the custom language, the representation of CPU components (registers, memory), and the entire fetch-decode-execute cycle must be implemented from scratch.\\n\\nIn the given repo, there are unit tests under `/tests`, and you can run via `bash tests/1.2.1.sh`.\\nHowever, you should NEVER first read and run the unit tests when you start with the task.\\nFirst directly read the source code and finish with the implementation without accessing the unit tests. During this process, you can write some your own test code or scripts to verify your implementation, but DO NOT read the content of the given tests.\\nAfter you think you have finished your implementation, you can run and read the provided unit tests to further verify and also correct the issues of your implementation.\\n\\nPassing all the tests of 1.2.1 and also all tests for previous tasks means you have finished this task.\\nYou MUST NOT modify any content under `/tests`.\\nYou also do not need to proceed to next tasks.\\n--------------------- END OF NEW TASK DESCRIPTION ---------------------\\n\\nPLEASE follow the format strictly! PLEASE EMIT ONE AND ONLY ONE FUNCTION CALL PER MESSAGE.\\n", "type": "text"}], "role": "user"}]'
EXPECTED_ASSISTANT_JSON = '{"content": [{"text": "I\'ll start by exploring the project directory to understand the current structure and then implement the CPU class as described in the task.", "type": "text"}], "role": "assistant", "tool_calls": [{"function": {"arguments": "{\\"command\\": \\"view\\", \\"path\\": \\"/workspace\\", \\"security_risk\\": \\"LOW\\", \\"summary\\": \\"I\'ll start by exploring the project directory to understand the current structure and then implement the CPU class as de\\"}", "name": "file_editor"}, "id": "call_000001_file_editor", "type": "function"}]}'
TOOL_SPECS_JSON = '[{"function": {"description": "View, create, and edit files in plain-text format.", "name": "file_editor", "parameters": {"properties": {"command": {"enum": ["view", "create", "str_replace", "insert", "undo_edit"], "type": "string"}, "file_text": {"description": "Content for create operations.", "type": "string"}, "insert_line": {"description": "Line before which to insert.", "type": "integer"}, "new_str": {"description": "Replacement text.", "type": "string"}, "old_str": {"description": "Text to replace.", "type": "string"}, "path": {"description": "Absolute file path.", "type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "view_range": {"description": "Optional inclusive line range for view.", "items": {"type": "integer"}, "type": "array"}}, "required": ["command", "path"], "type": "object"}}, "type": "function"}, {"function": {"description": "Signals completion of the current task or conversation.", "name": "finish", "parameters": {"properties": {"message": {"description": "Final message to send to the user.", "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["message"], "type": "object"}}, "type": "function"}, {"function": {"description": "Create, update, and inspect the task list for the current conversation.", "name": "task_tracker", "parameters": {"properties": {"command": {"enum": ["plan", "update", "view"], "type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "task_list": {"description": "Markdown task list content.", "type": "string"}}, "required": [], "type": "object"}}, "type": "function"}, {"function": {"description": "Execute a shell command in the terminal within a persistent shell session.", "name": "terminal", "parameters": {"properties": {"command": {"description": "The shell command or terminal input.", "type": "string"}, "is_input": {"description": "Whether command should be sent as input to a running process.", "type": "boolean"}, "reset": {"description": "Whether to reset the terminal session.", "type": "boolean"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "timeout": {"description": "Optional timeout in seconds.", "type": "number"}}, "required": ["command"], "type": "object"}}, "type": "function"}, {"function": {"description": "Log a thought without obtaining new information or changing the environment.", "name": "think", "parameters": {"properties": {"summary": {"description": "Concise summary of what this action does.", "type": "string"}, "thought": {"description": "The thought to log.", "type": "string"}}, "required": ["thought"], "type": "object"}}, "type": "function"}]'


class DatasetToolObservation(Observation):
    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if self.content:
            return self.content
        return [TextContent(text="Dataset validation tool result.")]


class DatasetToolExecutor(ToolExecutor):
    def __call__(self, action: Action, conversation=None) -> DatasetToolObservation:  # noqa: ARG002
        return DatasetToolObservation.from_text(
            f"Executed {action.__class__.__name__} for {DATASET_NAME}."
        )


class DatasetTool(ToolDefinition):
    @classmethod
    def create(cls, *args, **kwargs) -> list[Self]:
        return []


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def class_name(name: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", name)
    text = "".join(part[:1].upper() + part[1:] for part in parts if part)
    if not text or text[0].isdigit():
        text = "Dataset" + text
    return text


def sdk_content(content: Any) -> list[TextContent | ImageContent]:
    if content is None:
        return []
    if isinstance(content, str):
        return [TextContent(text=content)]
    blocks: list[TextContent | ImageContent] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            blocks.append(TextContent(text=item.get("text", "")))
        elif item.get("type") in {"image", "image_url"}:
            url = item.get("image_url", {}).get("url") if isinstance(item.get("image_url"), dict) else None
            if url:
                blocks.append(ImageContent(image_urls=[url]))
    return blocks


def sdk_message(raw: dict) -> Message:
    tool_calls = None
    if raw.get("tool_calls"):
        tool_calls = [
            MessageToolCall(
                id=tool_call["id"],
                name=tool_call["function"]["name"],
                arguments=tool_call["function"].get("arguments") or "{}",
                origin="completion",
            )
            for tool_call in raw["tool_calls"]
        ]
    return Message(
        role=raw["role"],
        content=sdk_content(raw.get("content")),
        tool_calls=tool_calls,
        tool_call_id=raw.get("tool_call_id"),
        name=raw.get("name"),
        reasoning_content=raw.get("reasoning_content"),
    )


def native_tool(name: str) -> ToolDefinition | None:
    if name == "finish":
        return FinishTool.create()[0]
    if name == "think":
        return ThinkTool.create()[0]
    if name == "terminal":
        return TerminalTool(
            description=UNIX_TOOL_DESCRIPTION,
            action_type=TerminalAction,
            observation_type=TerminalObservation,
            annotations=ToolAnnotations(
                title="terminal",
                readOnlyHint=False,
                destructiveHint=True,
                idempotentHint=False,
                openWorldHint=True,
            ),
        )
    if name == "file_editor":
        return FileEditorTool(
            description=FILE_EDITOR_DESCRIPTION,
            action_type=FileEditorAction,
            observation_type=FileEditorObservation,
            annotations=ToolAnnotations(
                title="file_editor",
                readOnlyHint=False,
                destructiveHint=True,
                idempotentHint=False,
                openWorldHint=False,
            ),
        )
    if name == "task_tracker":
        return TaskTrackerTool(
            description=TASK_TRACKER_DESCRIPTION,
            action_type=TaskTrackerAction,
            observation_type=TaskTrackerObservation,
            annotations=ToolAnnotations(
                title="task_tracker",
                readOnlyHint=False,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
        )
    browser_tools: dict[str, tuple[type[ToolDefinition], type[Action], str, bool]] = {
        "browser_navigate": (BrowserNavigateTool, BrowserNavigateAction, BROWSER_NAVIGATE_DESCRIPTION, False),
        "browser_click": (BrowserClickTool, BrowserClickAction, BROWSER_CLICK_DESCRIPTION, False),
        "browser_type": (BrowserTypeTool, BrowserTypeAction, BROWSER_TYPE_DESCRIPTION, False),
        "browser_get_state": (BrowserGetStateTool, BrowserGetStateAction, BROWSER_GET_STATE_DESCRIPTION, True),
        "browser_get_content": (BrowserGetContentTool, BrowserGetContentAction, BROWSER_GET_CONTENT_DESCRIPTION, True),
        "browser_scroll": (BrowserScrollTool, BrowserScrollAction, BROWSER_SCROLL_DESCRIPTION, False),
        "browser_go_back": (BrowserGoBackTool, BrowserGoBackAction, BROWSER_GO_BACK_DESCRIPTION, False),
    }
    if name in browser_tools:
        tool_cls, action_type, description, read_only = browser_tools[name]
        return tool_cls(
            description=description,
            action_type=action_type,
            observation_type=BrowserObservation,
            annotations=ToolAnnotations(
                title=name,
                readOnlyHint=read_only,
                destructiveHint=False,
                idempotentHint=read_only,
                openWorldHint=True,
            ),
        )
    return None


def custom_tool(spec: dict) -> ToolDefinition:
    function = spec["function"]
    name = function["name"]
    parameters = function.get("parameters") or {"type": "object", "properties": {}}
    action_type = Action.from_mcp_schema(f"{class_name(name)}Action", parameters)
    tool_cls = type(f"{class_name(name)}Tool", (DatasetTool,), {"name": name})
    return tool_cls(
        description=function.get("description") or f"Dataset-specific tool {name}.",
        action_type=action_type,
        observation_type=DatasetToolObservation,
        executor=DatasetToolExecutor(),
    )


def validation_tools() -> list[ToolDefinition]:
    tools: list[ToolDefinition] = []
    seen: set[str] = set()
    for spec in json.loads(TOOL_SPECS_JSON):
        function = spec.get("function") or {}
        name = function.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        tools.append(native_tool(name) or custom_tool(spec))
    return tools


def latest_log(log_dir: Path) -> Path:
    logs = sorted(log_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
    if not logs:
        raise RuntimeError("SDK did not write a completion log")
    return logs[-1]


def main() -> None:
    root = repo_root()
    load_env_file(root / ".env")
    load_env_file(Path.home() / "work" / "agent-data-protocol" / ".env")
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is required")

    prefix_messages = [sdk_message(message) for message in json.loads(PREFIX_MESSAGES_JSON)]
    log_dir = Path(tempfile.mkdtemp(prefix=f"{DATASET_NAME}-completion-"))
    llm = LLM(
        model=MODEL,
        api_key=SecretStr(api_key),
        base_url=os.getenv("LLM_BASE_URL"),
        log_completions=True,
        log_completions_folder=str(log_dir),
        max_output_tokens=512,
    )
    llm.completion(
        messages=prefix_messages,
        tools=validation_tools(),
        add_security_risk_prediction=True,
        temperature=0,
    )
    shutil.copyfile(latest_log(log_dir), Path(__file__).with_name("completion.json"))


if __name__ == "__main__":
    main()
