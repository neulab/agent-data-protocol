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


DATASET_NAME = 'scale_swe_distilled'
MODEL = os.getenv("LLM_MODEL", "openhands/minimax-m2.7")
RECORD_ID = 'fsspec_filesystem_spec_pr507_0'
ASSISTANT_INDEX = 2
PREFIX_MESSAGES_JSON = '[{"content": [{"text": "You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.\\n\\n<ROLE>\\n* Your primary role is to assist users by executing commands, modifying code, and solving technical problems effectively. You should be thorough, methodical, and prioritize quality over speed.\\n* If the user asks a question, like \\"why is X happening\\", don\'t try to fix the problem. Just give an answer to the question.\\n</ROLE>\\n\\n<MEMORY>\\n* Use `AGENTS.md` under the repository root as your persistent memory for repository-specific knowledge and context.\\n* Add important insights, patterns, and learnings to this file to improve future task performance.\\n* This repository skill is automatically loaded for every conversation and helps maintain context across sessions.\\n</MEMORY>\\n\\n<EFFICIENCY>\\n* Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action, e.g. combine multiple bash commands into one, using sed and grep to edit/view multiple files at once.\\n* When exploring the codebase, use efficient tools like find, grep, and git commands with appropriate filters to minimize unnecessary operations.\\n</EFFICIENCY>\\n\\n<FILE_SYSTEM_GUIDELINES>\\n* When a user provides a file path, do NOT assume it\'s relative to the current working directory. First explore the file system to locate the file before working on it.\\n* If asked to edit a file, edit the file directly, rather than creating a new file with a different filename.\\n* For global search-and-replace operations, consider using `sed` instead of opening file editors multiple times.\\n* NEVER create multiple versions of the same file with different suffixes. Modify the original file directly when making changes.\\n</FILE_SYSTEM_GUIDELINES>\\n\\n<CODE_QUALITY>\\n* Write clean, efficient code with minimal comments.\\n* Before implementing changes, first thoroughly understand the codebase through exploration.\\n* Place imports at the top of the file unless doing so would cause circular imports or other real issues.\\n</CODE_QUALITY>\\n\\n<VERSION_CONTROL>\\n* Exercise caution with git operations. Do NOT make potentially dangerous changes unless explicitly asked to do so.\\n* When committing changes, use `git status` to see all modified files, and stage all files necessary for the commit.\\n* Do NOT commit files that typically should not go into version control unless explicitly instructed by the user.\\n</VERSION_CONTROL>\\n\\n<PROBLEM_SOLVING_WORKFLOW>\\n1. EXPLORATION: Thoroughly explore relevant files and understand the context before proposing solutions.\\n2. ANALYSIS: Consider multiple approaches and select the most promising one.\\n3. TESTING: For bug fixes, create tests to verify issues before implementing fixes when appropriate.\\n4. IMPLEMENTATION: Make focused, minimal changes to address the problem.\\n5. VERIFICATION: Test your implementation thoroughly, including edge cases.\\n</PROBLEM_SOLVING_WORKFLOW>\\n\\n<SECURITY_RISK_ASSESSMENT>\\nWhen using tools that support the security_risk parameter, assess the safety risk of your actions:\\n* LOW: Safe, read-only actions.\\n* MEDIUM: Project-scoped edits or execution.\\n* HIGH: System-level or untrusted operations, privilege changes, or sensitive-data movement.\\n</SECURITY_RISK_ASSESSMENT>\\n\\n<EXTERNAL_SERVICES>\\n* When interacting with external services like GitHub, GitLab, or Bitbucket, use their respective APIs instead of browser-based interactions whenever possible.\\n* Only resort to browser-based interactions with these services if specifically requested by the user or if the required operation cannot be performed via API.\\n</EXTERNAL_SERVICES>\\n\\n<ENVIRONMENT_SETUP>\\n* When user asks you to run an application, do not stop if the application is not installed. Instead, install the application and run the command again when possible.\\n* If you encounter missing dependencies, first look around in the repository for existing dependency files and use them to install dependencies.\\n</ENVIRONMENT_SETUP>\\n", "type": "text"}], "role": "system"}, {"content": [{"text": "<uploaded_files>\\n/workspace/filesystem_spec\\n</uploaded_files>\\n\\nI\'ve uploaded a python code repository in the directory /workspace/filesystem_spec. Consider the following issue description:\\n\\n<issue_description>\\n# AbstractFileSystem.find includes directories in result due to variable shadowing\\n\\n## Description\\nI\'ve observed an issue where `AbstractFileSystem.find()` can incorrectly add one of the found directories to the returned files listing.\\n\\nThis appears to be due to a variable naming conflict within the `find` method. The loop variable used to iterate over `self.walk()` results seems to shadow the input `path` argument. Consequently, after the loop completes, the `path` variable holds the value of the last visited directory rather than the original input path. If the filesystem implementation returns `True` for `isfile()` on that directory (or if subsequent logic relies on `path`), that directory is erroneously appended to the output.\\n\\n## Reproduction Steps\\nHere is a standalone reproduction script that mocks a filesystem to demonstrate this behavior. It forces the condition where the last visited directory is treated as a valid file check target due to the variable shadowing.\\n\\n```python\\nimport sys\\nfrom fsspec.spec import AbstractFileSystem\\n\\nclass BugManifestingFileSystem(AbstractFileSystem):\\n    \\"\\"\\"\\n    A file system that manifests the variable shadowing bug.\\n    \\"\\"\\"\\n    \\n    def __init__(self, *args, **kwargs):\\n        super().__init__(*args, **kwargs)\\n        \\n    def ls(self, path, detail=True, **kwargs):\\n        # Mock directory structure\\n        if path == \\"root\\":\\n            return [\\n                {\\"name\\": \\"root/dir1\\", \\"type\\": \\"directory\\"},\\n                {\\"name\\": \\"root/dir2\\", \\"type\\": \\"directory\\"},\\n                {\\"name\\": \\"root/file1.txt\\", \\"type\\": \\"file\\"},\\n            ]\\n        elif path == \\"root/dir1\\":\\n            return [{\\"name\\": \\"root/dir1/file2.txt\\", \\"type\\": \\"file\\"}]\\n        elif path == \\"root/dir2\\":\\n            return [{\\"name\\": \\"root/dir2/file3.txt\\", \\"type\\": \\"file\\"}]\\n        else:\\n            raise FileNotFoundError(f\\"Path not found: {path}\\")\\n    \\n    def isfile(self, path):\\n        # Return True for dir2 to manifest the bug.\\n        # This simulates a scenario where the shadowed path variable \\n        # (holding the last visited directory) is checked.\\n        return path == \\"root/dir2\\"\\n    \\n    def isdir(self, path):\\n        return path in [\\"root\\", \\"root/dir1\\", \\"root/dir2\\"]\\n    \\n    def _strip_protocol(self, path):\\n        return path\\n\\ndef test_find_shadowing():\\n    fs = BugManifestingFileSystem()\\n    \\n    # We expect find() to return only files.\\n    # However, due to shadowing, \'root/dir2\' (the last directory processed)\\n    # is checked against isfile() at the end of the method using the shadowed variable.\\n    result = fs.find(\\"root\\")\\n    \\n    print(f\\"Found: {result}\\")\\n    \\n    # Check that the directory is NOT in the results\\n    if \\"root/dir2\\" in result:\\n        print(\\"FAILURE: \'root/dir2\' (a directory) was found in the result list.\\")\\n        sys.exit(1)\\n        \\n    expected = [\\"root/file1.txt\\", \\"root/dir1/file2.txt\\", \\"root/dir2/file3.txt\\"]\\n    if sorted(result) != sorted(expected):\\n        print(f\\"FAILURE: Result mismatch.\\\\nExpected: {expected}\\\\nGot: {result}\\")\\n        sys.exit(1)\\n        \\n    print(\\"SUCCESS: Bug not present.\\")\\n\\nif __name__ == \\"__main__\\":\\n    test_find_shadowing()\\n```\\n\\n## Expected Behavior\\nThe `find` method should return only the files found in the directory tree:\\n`[\'root/file1.txt\', \'root/dir1/file2.txt\', \'root/dir2/file3.txt\']`\\n\\n## Actual Behavior\\nThe `find` method returns the files plus the last visited directory:\\n`[\'root/file1.txt\', \'root/dir1/file2.txt\', \'root/dir2/file3.txt\', \'root/dir2\']`\\n</issue_description>\\n\\nCan you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?\\nI\'ve already taken care of all changes to any of the test files described in the <issue_description>. This means you DON\'T have to modify the testing logic or any of the tests in any way!\\nAlso the development Python environment is already set up for you (i.e., all dependencies already installed), so you don\'t need to install other packages.\\nYour task is to make the minimal changes to non-test files in the /workspace/filesystem_spec directory to ensure the <issue_description> is satisfied.\\n\\nFollow these phases to resolve the issue:\\n\\nPhase 1. READING: read the problem and reword it in clearer terms\\n   1.1 If there are code or config snippets. Express in words any best practices or conventions in them.\\n   1.2 Highlight message errors, method names, variables, file names, stack traces, and technical details.\\n   1.3 Explain the problem in clear terms.\\n   1.4 Enumerate the steps to reproduce the problem.\\n   1.5 Highlight any best practices to take into account when testing and fixing the issue\\n\\nPhase 2. RUNNING: install and run the tests on the repository\\n   2.1 Follow the readme\\n   2.2 Install the environment and anything needed\\n   2.2 Iterate and figure out how to run the tests\\n\\nPhase 3. EXPLORATION: find the files that are related to the problem and possible solutions\\n   3.1 Use `grep` to search for relevant methods, classes, keywords and error messages.\\n   3.2 Identify all files related to the problem statement.\\n   3.3 Propose the methods and files to fix the issue and explain why.\\n   3.4 From the possible file locations, select the most likely location to fix the issue.\\n\\nPhase 4. TEST CREATION: before implementing any fix, create a script to reproduce and verify the issue.\\n   4.1 Look at existing test files in the repository to understand the test format/structure.\\n   4.2 Create a minimal reproduction script that reproduces the located issue.\\n   4.3 Run the reproduction script to confirm you are reproducing the issue.\\n   4.4 Adjust the reproduction script as necessary.\\n\\nPhase 5. FIX ANALYSIS: state clearly the problem and how to fix it\\n   5.1 State clearly what the problem is.\\n   5.2 State clearly where the problem is located.\\n   5.3 State clearly how the test reproduces the issue.\\n   5.4 State clearly the best practices to take into account in the fix.\\n   5.5 State clearly how to fix the problem.\\n\\nPhase 6. FIX IMPLEMENTATION: Edit the source code to implement your chosen solution.\\n   6.1 Make minimal, focused changes to fix the issue.\\n\\nPhase 7. VERIFICATION: Test your implementation thoroughly.\\n   7.1 Run your reproduction script to verify the fix works.\\n   7.2 Add edge cases to your test script to ensure comprehensive coverage.\\n   7.3 Run existing tests related to the modified code to ensure you haven\'t broken anything.\\n\\n8. FINAL REVIEW: Carefully re-read the problem description and compare your changes with the base commit 0b39f03d3666dcc02d662269e75dda25cce332ed.\\n   8.1 Ensure you\'ve fully addressed all requirements.\\n   8.2 Run any tests in the repository related to:\\n     8.2.1 The issue you are fixing\\n     8.2.2 The files you modified\\n     8.2.3 The functions you changed\\n   8.3 If any tests fail, revise your implementation until all tests pass\\n\\nBe thorough in your exploration, testing, and reasoning. It\'s fine if your thinking process is lengthy - quality and completeness are more important than brevity.\\n--------------------- END OF NEW TASK DESCRIPTION ---------------------\\n\\nPLEASE follow the format strictly! PLEASE EMIT ONE AND ONLY ONE FUNCTION CALL PER MESSAGE.\\n", "type": "text"}], "role": "user"}]'
EXPECTED_ASSISTANT_JSON = '{"content": [{"text": "I\'ll start with Phase 1: READING.\\n\\nThe issue is about a variable shadowing bug in `AbstractFileSystem.find()` method. The loop variable used to iterate over `self.walk()` results shadows the input `path` argument. After the loop, `path` holds the last visited directory rather than the original input path. If the filesystem\'s `isfile()` returns True for that directory (or other logic relies on `path`), that directory is incorrectly appended to the output.\\n\\nThe reproduction script demonstrates a mock filesystem where `isfile(\\"root/dir2\\")` returns True, causing the directory to be added. The expected behavior is that `find` returns only files, not directories.\\n\\nBest practices: avoid variable shadowing, preserve original function arguments, ensure loop variables don\'t overwrite important parameters.\\n\\nNow, let\'s proceed to Phase 2: RUNNING. First, explore the repository to understand its structure and how to run tests.", "type": "text"}], "role": "assistant", "tool_calls": [{"function": {"arguments": "{\\"command\\": \\"view\\", \\"path\\": \\"/workspace/filesystem_spec\\", \\"security_risk\\": \\"LOW\\", \\"summary\\": \\"I\'ll start with Phase 1: READING. The issue is about a variable shadowing bug in `AbstractFileSystem.find()` method. The\\"}", "name": "file_editor"}, "id": "call_000001_file_editor", "type": "function"}]}'
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
