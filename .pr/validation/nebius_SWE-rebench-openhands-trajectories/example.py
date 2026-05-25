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


DATASET_NAME = 'nebius_SWE-rebench-openhands-trajectories'
MODEL = os.getenv("LLM_MODEL", "openhands/minimax-m2.7")
RECORD_ID = 'chatcmpl-eb27c5fe3baa10d5272928b4e016d84f'
ASSISTANT_INDEX = 2
PREFIX_MESSAGES_JSON = '[{"content": [{"text": "You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.\\n\\n<ROLE>\\n* Your primary role is to assist users by executing commands, modifying code, and solving technical problems effectively. You should be thorough, methodical, and prioritize quality over speed.\\n* If the user asks a question, like \\"why is X happening\\", don\'t try to fix the problem. Just give an answer to the question.\\n</ROLE>\\n\\n<MEMORY>\\n* Use `AGENTS.md` under the repository root as your persistent memory for repository-specific knowledge and context.\\n* Add important insights, patterns, and learnings to this file to improve future task performance.\\n* This repository skill is automatically loaded for every conversation and helps maintain context across sessions.\\n</MEMORY>\\n\\n<EFFICIENCY>\\n* Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action, e.g. combine multiple bash commands into one, using sed and grep to edit/view multiple files at once.\\n* When exploring the codebase, use efficient tools like find, grep, and git commands with appropriate filters to minimize unnecessary operations.\\n</EFFICIENCY>\\n\\n<FILE_SYSTEM_GUIDELINES>\\n* When a user provides a file path, do NOT assume it\'s relative to the current working directory. First explore the file system to locate the file before working on it.\\n* If asked to edit a file, edit the file directly, rather than creating a new file with a different filename.\\n* For global search-and-replace operations, consider using `sed` instead of opening file editors multiple times.\\n* NEVER create multiple versions of the same file with different suffixes. Modify the original file directly when making changes.\\n</FILE_SYSTEM_GUIDELINES>\\n\\n<CODE_QUALITY>\\n* Write clean, efficient code with minimal comments.\\n* Before implementing changes, first thoroughly understand the codebase through exploration.\\n* Place imports at the top of the file unless doing so would cause circular imports or other real issues.\\n</CODE_QUALITY>\\n\\n<VERSION_CONTROL>\\n* Exercise caution with git operations. Do NOT make potentially dangerous changes unless explicitly asked to do so.\\n* When committing changes, use `git status` to see all modified files, and stage all files necessary for the commit.\\n* Do NOT commit files that typically should not go into version control unless explicitly instructed by the user.\\n</VERSION_CONTROL>\\n\\n<PROBLEM_SOLVING_WORKFLOW>\\n1. EXPLORATION: Thoroughly explore relevant files and understand the context before proposing solutions.\\n2. ANALYSIS: Consider multiple approaches and select the most promising one.\\n3. TESTING: For bug fixes, create tests to verify issues before implementing fixes when appropriate.\\n4. IMPLEMENTATION: Make focused, minimal changes to address the problem.\\n5. VERIFICATION: Test your implementation thoroughly, including edge cases.\\n</PROBLEM_SOLVING_WORKFLOW>\\n\\n<SECURITY_RISK_ASSESSMENT>\\nWhen using tools that support the security_risk parameter, assess the safety risk of your actions:\\n* LOW: Safe, read-only actions.\\n* MEDIUM: Project-scoped edits or execution.\\n* HIGH: System-level or untrusted operations, privilege changes, or sensitive-data movement.\\n</SECURITY_RISK_ASSESSMENT>\\n\\n<EXTERNAL_SERVICES>\\n* When interacting with external services like GitHub, GitLab, or Bitbucket, use their respective APIs instead of browser-based interactions whenever possible.\\n* Only resort to browser-based interactions with these services if specifically requested by the user or if the required operation cannot be performed via API.\\n</EXTERNAL_SERVICES>\\n\\n<ENVIRONMENT_SETUP>\\n* When user asks you to run an application, do not stop if the application is not installed. Instead, install the application and run the command again when possible.\\n* If you encounter missing dependencies, first look around in the repository for existing dependency files and use them to install dependencies.\\n</ENVIRONMENT_SETUP>\\n", "type": "text"}], "role": "system"}, {"content": [{"text": "<uploaded_files>\\n/workspace/tianocore__edk2-pytool-library__0.15\\n</uploaded_files>\\n\\nI\'ve uploaded a python code repository in the directory tianocore__edk2-pytool-library__0.15. Consider the following issue description:\\n\\n<issue_description>\\n[Bug]: GetEdk2RelativePathFromAbsolutePath() Can Silently Return Non-POSIX Path\\n### Contact Details\\n\\n_No response_\\n\\n### Describe the Bug\\n\\nGetEdk2RelativePathFromAbsolutePath() is currently defined as follows:\\r\\n\\r\\n```python\\r\\n        \\"\\"\\"Given an absolute path return a edk2 path relative to workspace or packagespath.\\r\\n\\r\\n        Note: absolute path must be in the OS specific path form\\r\\n        Note: the relative path will be in POSIX-like path form\\r\\n\\r\\n        Args:\\r\\n            abspath (os.PathLike): absolute path to a file or directory. Path must contain OS specific separator.\\r\\n\\r\\n        Returns:\\r\\n            (os.PathLike): POSIX-like relative path to workspace or packagespath\\r\\n            (None): abspath is none\\r\\n            (None): path is not valid\\r\\n        \\"\\"\\"\\r\\n```\\r\\n\\r\\nIt requires an OS-specific path (to the current OS) to be given as input. However, it does not return an exception or other fail in a well-defined way if path given does not conform to the current OS. It likely returns a non-POSIX path.\\r\\n\\r\\nSo, this is not a bug with the function in the case the user passes the expected path format but a bug in validating the function input values.\\n\\n### Reproduction steps\\n\\nThe following code demonstrates the problem if run on Linux since it passes a Windows style path to the function.\\r\\n\\r\\n```\\r\\n        p = f\\"{ws_pkg_abs}\\\\\\\\module_2\\\\\\\\X64\\\\\\\\TestFile.inf\\"\\r\\n        expected_rel_from_abs_path = PurePath(os.path.join(ws_p_name, \\"module_2\\", \\"X64\\", \\"TestFile.inf\\")).as_posix()\\r\\n        actual_rel_from_abs_path = pathobj.GetEdk2RelativePathFromAbsolutePath(p)\\r\\n>       self.assertEqual(expected_rel_from_abs_path, actual_rel_from_abs_path)\\r\\nE       AssertionError: \'WSTestPkg/module_2/X64/TestFile.inf\' != \'WSTestPkg\\\\\\\\module_2\\\\\\\\X64\\\\\\\\TestFile.inf\'\\r\\nE       - WSTestPkg/module_2/X64/TestFile.inf\\r\\nE       ?          ^        ^   ^\\r\\nE       + WSTestPkg\\\\module_2\\\\X64\\\\TestFile.inf\\r\\n```\\n\\n### Expected behavior\\n\\nThe function should ensure that it handles input in a predictable and well-defined manner.\\n\\n### Execution Environment\\n\\n_No response_\\n\\n### Pip packages\\n\\n_No response_\\n\\n### Additional context\\n\\n_No response_\\n</issue_description>\\n\\nCan you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?\\nI\'ve already taken care of all changes to any of the test files described in the <issue_description>. This means you DON\'T have to modify the testing logic or any of the tests in any way!\\nAlso the development Python environment is already set up for you (i.e., all dependencies already installed), so you don\'t need to install other packages.\\nYour task is to make the minimal changes to non-test files in the /workspace/tianocore__edk2-pytool-library__0.15 directory to ensure the <issue_description> is satisfied.\\n\\nFollow these phases to resolve the issue:\\n\\nPhase 1. READING: read the problem and reword it in clearer terms\\n   1.1 If there are code or config snippets. Express in words any best practices or conventions in them.\\n   1.2 Hightlight message errors, method names, variables, file names, stack traces, and technical details.\\n   1.3 Explain the problem in clear terms.\\n   1.4 Enumerate the steps to reproduce the problem.\\n   1.5 Hightlight any best practices to take into account when testing and fixing the issue\\n\\nPhase 2. RUNNING: install and run the tests on the repository\\n   2.1 Follow the readme\\n   2.2 Install the environment and anything needed\\n   2.2 Iterate and figure out how to run the tests\\n\\nPhase 3. EXPLORATION: find the files that are related to the problem and possible solutions\\n   3.1 Use `grep` to search for relevant methods, classes, keywords and error messages.\\n   3.2 Identify all files related to the problem statement.\\n   3.3 Propose the methods and files to fix the issue and explain why.\\n   3.4 From the possible file locations, select the most likely location to fix the issue.\\n\\nPhase 4. TEST CREATION: before implementing any fix, create a script to reproduce and verify the issue.\\n   4.1 Look at existing test files in the repository to understand the test format/structure.\\n   4.2 Create a minimal reproduction script that reproduces the located issue.\\n   4.3 Run the reproduction script to confirm you are reproducing the issue.\\n   4.4 Adjust the reproduction script as necessary.\\n\\nPhase 5. FIX ANALYSIS: state clearly the problem and how to fix it\\n   5.1 State clearly what the problem is.\\n   5.2 State clearly where the problem is located.\\n   5.3 State clearly how the test reproduces the issue.\\n   5.4 State clearly the best practices to take into account in the fix.\\n   5.5 State clearly how to fix the problem.\\n\\nPhase 6. FIX IMPLEMENTATION: Edit the source code to implement your chosen solution.\\n   6.1 Make minimal, focused changes to fix the issue.\\n\\nPhase 7. VERIFICATION: Test your implementation thoroughly.\\n   7.1 Run your reproduction script to verify the fix works.\\n   7.2 Add edge cases to your test script to ensure comprehensive coverage.\\n   7.3 Run existing tests related to the modified code to ensure you haven\'t broken anything.\\n\\n8. FINAL REVIEW: Carefully re-read the problem description and compare your changes with the base commit f521d59041afee6a8a82206b3871960409aaa612.\\n   8.1 Ensure you\'ve fully addressed all requirements.\\n   8.2 Run any tests in the repository related to:\\n     8.2.1 The issue you are fixing\\n     8.2.2 The files you modified\\n     8.2.3 The functions you changed\\n   8.3 If any tests fail, revise your implementation until all tests pass\\n\\nBe thorough in your exploration, testing, and reasoning. It\'s fine if your thinking process is lengthy - quality and completeness are more important than brevity.", "type": "text"}], "role": "user"}]'
EXPECTED_ASSISTANT_JSON = '{"content": [{"text": "I\'ll help you implement the necessary changes to fix the issue with `GetEdk2RelativePathFromAbsolutePath()` silently returning non-POSIX paths. Let me start by following the phases you outlined.\\n\\n## Phase 1. READING: Understanding the Problem", "type": "text"}], "role": "assistant", "tool_calls": [{"function": {"arguments": "{\\"thought\\": \\"Let me carefully read and understand the issue:\\\\n\\\\n1.1 Code/config snippets analysis:\\\\n- The function `GetEdk2RelativePathFromAbsolutePath()` is supposed to take an OS-specific absolute path and return a POSIX-like relative path\\\\n- The docstring says: \\\\\\"Note: absolute path must be in the OS specific path form\\\\\\" and \\\\\\"Note: the relative path will be in POSIX-like path form\\\\\\"\\\\n- The function signature expects `abspath (os.PathLike)` with OS specific separators\\\\n\\\\n1.2 Technical details:\\\\n- Method name: `GetEdk2RelativePathFromAbsolutePath()`\\\\n- Error message shows: `\'WSTestPkg/module_2/X64/TestFile.inf\' != \'WSTestPkg\\\\\\\\\\\\\\\\module_2\\\\\\\\\\\\\\\\X64\\\\\\\\\\\\\\\\TestFile.inf\'`\\\\n- The expected result uses forward slashes (POSIX) but actual result shows backslashes (Windows)\\\\n- Variable names: `expected_rel_from_abs_path`, `actual_rel_from_abs_path`, `ws_pkg_abs`, `ws_p_name`\\\\n\\\\n1.3 Problem explanation:\\\\nThe function is designed to convert OS-specific absolute paths to POSIX-like relative paths. However, when given a path that doesn\'t conform to the current OS format (e.g., passing a Windows-style path with backslashes on Linux), it doesn\'t validate the input properly and can return a non-POSIX path instead of handling it correctly or raising an appropriate error.\\\\n\\\\n1.4 Steps to reproduce:\\\\n1. Run on Linux system\\\\n2. Pass a Windows-style path (with backslashes) to `GetEdk2RelativePathFromAbsolutePath()`\\\\n3. The function returns a path with backslashes instead of converting to POSIX format\\\\n\\\\n1.5 Best practices for testing/fixing:\\\\n- Input validation: Check if the input path conforms to expected OS format\\\\n- Consistent output: Always return POSIX-like paths regardless of input inconsistencies\\\\n- Clear error handling: Either normalize the input or raise appropriate exceptions\\\\n- Backward compatibility: Don\'t break existing functionality for valid inputs\\", \\"summary\\": \\"I\'ll help you implement the necessary changes to fix the issue with `GetEdk2RelativePathFromAbsolutePath()` silently ret\\"}", "name": "think"}, "id": "call_000001_think", "type": "function"}]}'
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
