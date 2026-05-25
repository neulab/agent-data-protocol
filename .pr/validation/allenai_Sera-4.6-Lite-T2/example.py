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


DATASET_NAME = 'allenai_Sera-4.6-Lite-T2'
MODEL = os.getenv("LLM_MODEL", "openhands/minimax-m2.7")
RECORD_ID = 'tinydb_10309'
ASSISTANT_INDEX = 2
PREFIX_MESSAGES_JSON = '[{"content": [{"text": "You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.\\n\\n<ROLE>\\n* Your primary role is to assist users by executing commands, modifying code, and solving technical problems effectively. You should be thorough, methodical, and prioritize quality over speed.\\n* If the user asks a question, like \\"why is X happening\\", don\'t try to fix the problem. Just give an answer to the question.\\n</ROLE>\\n\\n<MEMORY>\\n* Use `AGENTS.md` under the repository root as your persistent memory for repository-specific knowledge and context.\\n* Add important insights, patterns, and learnings to this file to improve future task performance.\\n* This repository skill is automatically loaded for every conversation and helps maintain context across sessions.\\n</MEMORY>\\n\\n<EFFICIENCY>\\n* Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action, e.g. combine multiple bash commands into one, using sed and grep to edit/view multiple files at once.\\n* When exploring the codebase, use efficient tools like find, grep, and git commands with appropriate filters to minimize unnecessary operations.\\n</EFFICIENCY>\\n\\n<FILE_SYSTEM_GUIDELINES>\\n* When a user provides a file path, do NOT assume it\'s relative to the current working directory. First explore the file system to locate the file before working on it.\\n* If asked to edit a file, edit the file directly, rather than creating a new file with a different filename.\\n* For global search-and-replace operations, consider using `sed` instead of opening file editors multiple times.\\n* NEVER create multiple versions of the same file with different suffixes. Modify the original file directly when making changes.\\n</FILE_SYSTEM_GUIDELINES>\\n\\n<CODE_QUALITY>\\n* Write clean, efficient code with minimal comments.\\n* Before implementing changes, first thoroughly understand the codebase through exploration.\\n* Place imports at the top of the file unless doing so would cause circular imports or other real issues.\\n</CODE_QUALITY>\\n\\n<VERSION_CONTROL>\\n* Exercise caution with git operations. Do NOT make potentially dangerous changes unless explicitly asked to do so.\\n* When committing changes, use `git status` to see all modified files, and stage all files necessary for the commit.\\n* Do NOT commit files that typically should not go into version control unless explicitly instructed by the user.\\n</VERSION_CONTROL>\\n\\n<PROBLEM_SOLVING_WORKFLOW>\\n1. EXPLORATION: Thoroughly explore relevant files and understand the context before proposing solutions.\\n2. ANALYSIS: Consider multiple approaches and select the most promising one.\\n3. TESTING: For bug fixes, create tests to verify issues before implementing fixes when appropriate.\\n4. IMPLEMENTATION: Make focused, minimal changes to address the problem.\\n5. VERIFICATION: Test your implementation thoroughly, including edge cases.\\n</PROBLEM_SOLVING_WORKFLOW>\\n\\n<SECURITY_RISK_ASSESSMENT>\\nWhen using tools that support the security_risk parameter, assess the safety risk of your actions:\\n* LOW: Safe, read-only actions.\\n* MEDIUM: Project-scoped edits or execution.\\n* HIGH: System-level or untrusted operations, privilege changes, or sensitive-data movement.\\n</SECURITY_RISK_ASSESSMENT>\\n\\n<EXTERNAL_SERVICES>\\n* When interacting with external services like GitHub, GitLab, or Bitbucket, use their respective APIs instead of browser-based interactions whenever possible.\\n* Only resort to browser-based interactions with these services if specifically requested by the user or if the required operation cannot be performed via API.\\n</EXTERNAL_SERVICES>\\n\\n<ENVIRONMENT_SETUP>\\n* When user asks you to run an application, do not stop if the application is not installed. Instead, install the application and run the command again when possible.\\n* If you encounter missing dependencies, first look around in the repository for existing dependency files and use them to install dependencies.\\n</ENVIRONMENT_SETUP>\\n", "type": "text"}], "role": "system"}, {"content": [{"text": "<uploaded_files>\\n/testbed\\n</uploaded_files>\\nI\'ve uploaded a python code repository in the directory /testbed. Consider the following PR description:\\n\\n<pr_description>\\n\\nmetadata:\\n  id: tinydb-jsonstorage-write-issue\\n  type: bug\\n  status: open\\n  priority: medium\\n  created: 2024-01-15T10:30:00.000\\n  updated: \\n\\nproject:\\n  repository: https://github.com/msiemens/tinydb\\n  component: storages\\n  version: \\n\\ntitle: JSONStorage.write() fails to write data in certain edge cases\\n\\ndescription:\\n  summary: |\\n    The JSONStorage.write() method contains an unnecessary file existence check that prevents writing in edge cases where the file might not exist at the exact moment of checking, even though a valid file handle is open.\\n  \\n  detailed: |\\n    The JSONStorage.write() implementation has a problematic check that verifies if the file exists using os.path.exists() before attempting to write. This check is redundant since we already have an open file handle, and it can cause write operations to fail in race conditions or on unreliable filesystems. When the file is deleted after the handle is opened but before the write operation, the existence check fails and no data is written, leading to silent data loss.\\n\\nreproduction:\\n  preconditions: |\\n    - Using JSONStorage for database persistence\\n    - File handle is opened successfully\\n    - File gets deleted or becomes inaccessible between opening and writing\\n  \\n  steps: |\\n    1. Create a JSONStorage instance with a file path\\n    2. Delete the file from the filesystem while the handle is still open\\n    3. Attempt to write data using the storage.write() method\\n    4. Observe that no data is written and no error is raised\\n  \\n  code_sample: |\\n    import os\\n    from tinydb.storages import JSONStorage\\n    \\n    # Create storage\\n    storage = JSONStorage(\'test.json\')\\n    \\n    # Delete file while handle is open (simulating race condition)\\n    os.unlink(\'test.json\')\\n    \\n    # Try to write - this silently fails\\n    data = {\\"test\\": {\\"1\\": {\\"key\\": \\"value\\"}}}\\n    storage.write(data)\\n    \\n    # Data is not written!\\n    read_data = storage.read()\\n    print(read_data)  # None or incorrect data\\n  \\n  input_data: |\\n    Any dictionary data to be written to JSON storage\\n\\nbehavior:\\n  actual: |\\n    The write operation silently fails when the file doesn\'t exist at the exact moment of the existence check, even though a valid file handle is open. This results in data not being persisted without any error indication.\\n  \\n  expected: |\\n    The write operation should successfully write data through the open file handle regardless of whether the file exists at the moment of checking, since the handle is already valid.\\n  \\n  error_output: |\\n    No error is raised, but data is not written to storage\\n\\nimpact:\\n  severity: Medium - can cause silent data loss in production\\n  affected_users: Users using JSONStorage in environments with potential race conditions or unreliable filesystems\\n  workaround: |\\n    Ensure files are not deleted while handles are open, or catch and retry write operations\\n\\nenvironment:\\n  os: All operating systems\\n  language_version: Python 3.x\\n  dependencies: |\\n    tinydb\\n  \\n  configuration: |\\n    Using JSONStorage with default settings\\n\\nanalysis:\\n  root_cause: |\\n    The JSONStorage.write() method includes an unnecessary os.path.exists(self._handle.name) check before writing, which can fail in edge cases even when the file handle is valid.\\n  \\n  affected_code: |\\n    tinydb/storages.py, lines 147-148 in the write() method\\n  \\n  related_issues: |\\n    None known\\n\\nproposed_solution:\\n  approach: |\\n    Remove the unnecessary os.path.exists() check and directly write to the open file handle. The file handle is already validated when opened, so the existence check is redundant and harmful.\\n  \\n  alternatives: |\\n    Could add retry logic or better error handling, but removing the check is the simplest and most correct solution.\\n  \\n  breaking_changes: No\\n  \\n  migration_notes: |\\n    No migration needed - this is a bug fix that improves reliability\\n\\ntesting:\\n  test_cases: |\\n    - Test writing to a file that gets deleted after handle is opened\\n    - Test writing on unreliable filesystems\\n    - Test normal write operations still work\\n    - Test with various JSON formatting options\\n  \\n  regression_risk: Low - the fix removes problematic code without changing the core write logic\\n\\nadditional_context:\\n  screenshots: |\\n    \\n  \\n  logs: |\\n    \\n  \\n  references: |\\n    None\\n  \\n  notes: |\\n    This issue can be particularly problematic in containerized environments or when using network filesystems where file existence might be transient.\\n\\n</pr_description>\\n\\nCan you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?\\nI\'ve already taken care of all changes to any of the test files described in the <pr_description>. This means you DON\'T have to modify the testing logic or any of the tests in any way!\\nYour task is to make the minimal changes to non-tests files in the /testbed directory to ensure the <pr_description> is satisfied.\\nFollow these steps to resolve the issue:\\n1. As a first step, it might be a good idea to find and read code relevant to the <pr_description>\\n2. Create a script to reproduce the error and execute it with `python <filename.py>` using the bash tool, to confirm the error\\n3. Edit the sourcecode of the repo to resolve the issue\\n4. Rerun your reproduce script and confirm that the error is fixed!\\n5. Think about edgecases and make sure your fix handles them as well\\nYour thinking should be thorough and so it\'s fine if it\'s very long.", "type": "text"}], "role": "user"}]'
EXPECTED_ASSISTANT_JSON = '{"content": [{"text": "I\'ll help you implement the necessary changes to fix the JSONStorage.write() issue. Let\'s start by exploring the repository structure and understanding the problem.", "type": "text"}], "reasoning_content": "Let me break down the problem:\\n\\n1. The issue is in the `JSONStorage.write()` method in `tinydb/storages.py`\\n2. There\'s an unnecessary `os.path.exists(self._handle.name)` check before writing\\n3. This check can fail in race conditions where the file is deleted after the handle is opened but before writing\\n4. The fix is to remove this unnecessary check since we already have a valid file handle\\n\\nLet me start by exploring the repository structure and finding the relevant code.", "role": "assistant", "tool_calls": [{"function": {"arguments": "{\\"command\\": \\"view\\", \\"path\\": \\"/testbed\\", \\"security_risk\\": \\"LOW\\", \\"summary\\": \\"I\'ll help you implement the necessary changes to fix the JSONStorage.write() issue. Let\'s start by exploring the reposit\\"}", "name": "file_editor"}, "id": "call_000001_file_editor", "type": "function"}]}'
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
