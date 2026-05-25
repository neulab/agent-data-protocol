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


DATASET_NAME = 'nemotron_terminal_corpus'
MODEL = os.getenv("LLM_MODEL", "openhands/minimax-m2.7")
RECORD_ID = 'data_processing_task_3216_episode-7'
ASSISTANT_INDEX = 2
PREFIX_MESSAGES_JSON = '[{"content": [{"text": "You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.\\n\\n<ROLE>\\n* Your primary role is to assist users by executing commands, modifying code, and solving technical problems effectively. You should be thorough, methodical, and prioritize quality over speed.\\n* If the user asks a question, like \\"why is X happening\\", don\'t try to fix the problem. Just give an answer to the question.\\n</ROLE>\\n\\n<MEMORY>\\n* Use `AGENTS.md` under the repository root as your persistent memory for repository-specific knowledge and context.\\n* Add important insights, patterns, and learnings to this file to improve future task performance.\\n* This repository skill is automatically loaded for every conversation and helps maintain context across sessions.\\n</MEMORY>\\n\\n<EFFICIENCY>\\n* Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action, e.g. combine multiple bash commands into one, using sed and grep to edit/view multiple files at once.\\n* When exploring the codebase, use efficient tools like find, grep, and git commands with appropriate filters to minimize unnecessary operations.\\n</EFFICIENCY>\\n\\n<FILE_SYSTEM_GUIDELINES>\\n* When a user provides a file path, do NOT assume it\'s relative to the current working directory. First explore the file system to locate the file before working on it.\\n* If asked to edit a file, edit the file directly, rather than creating a new file with a different filename.\\n* For global search-and-replace operations, consider using `sed` instead of opening file editors multiple times.\\n* NEVER create multiple versions of the same file with different suffixes. Modify the original file directly when making changes.\\n</FILE_SYSTEM_GUIDELINES>\\n\\n<CODE_QUALITY>\\n* Write clean, efficient code with minimal comments.\\n* Before implementing changes, first thoroughly understand the codebase through exploration.\\n* Place imports at the top of the file unless doing so would cause circular imports or other real issues.\\n</CODE_QUALITY>\\n\\n<VERSION_CONTROL>\\n* Exercise caution with git operations. Do NOT make potentially dangerous changes unless explicitly asked to do so.\\n* When committing changes, use `git status` to see all modified files, and stage all files necessary for the commit.\\n* Do NOT commit files that typically should not go into version control unless explicitly instructed by the user.\\n</VERSION_CONTROL>\\n\\n<PROBLEM_SOLVING_WORKFLOW>\\n1. EXPLORATION: Thoroughly explore relevant files and understand the context before proposing solutions.\\n2. ANALYSIS: Consider multiple approaches and select the most promising one.\\n3. TESTING: For bug fixes, create tests to verify issues before implementing fixes when appropriate.\\n4. IMPLEMENTATION: Make focused, minimal changes to address the problem.\\n5. VERIFICATION: Test your implementation thoroughly, including edge cases.\\n</PROBLEM_SOLVING_WORKFLOW>\\n\\n<SECURITY_RISK_ASSESSMENT>\\nWhen using tools that support the security_risk parameter, assess the safety risk of your actions:\\n* LOW: Safe, read-only actions.\\n* MEDIUM: Project-scoped edits or execution.\\n* HIGH: System-level or untrusted operations, privilege changes, or sensitive-data movement.\\n</SECURITY_RISK_ASSESSMENT>\\n\\n<EXTERNAL_SERVICES>\\n* When interacting with external services like GitHub, GitLab, or Bitbucket, use their respective APIs instead of browser-based interactions whenever possible.\\n* Only resort to browser-based interactions with these services if specifically requested by the user or if the required operation cannot be performed via API.\\n</EXTERNAL_SERVICES>\\n\\n<ENVIRONMENT_SETUP>\\n* When user asks you to run an application, do not stop if the application is not installed. Instead, install the application and run the command again when possible.\\n* If you encounter missing dependencies, first look around in the repository for existing dependency files and use them to install dependencies.\\n</ENVIRONMENT_SETUP>\\n", "type": "text"}], "role": "system"}, {"content": [{"text": "You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.\\n\\nFormat your response as JSON with the following structure:\\n\\n{\\n  \\"analysis\\": \\"Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?\\",\\n  \\"plan\\": \\"Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.\\",\\n  \\"commands\\": [\\n    {\\n      \\"keystrokes\\": \\"ls -la\\\\n\\",\\n      \\"duration\\": 0.1\\n    },\\n    {\\n      \\"keystrokes\\": \\"cd project\\\\n\\",\\n      \\"duration\\": 0.1\\n    }\\n  ],\\n  \\"task_complete\\": true\\n}\\n\\nRequired fields:\\n- \\"analysis\\": Your analysis of the current situation\\n- \\"plan\\": Your plan for the next steps\\n- \\"commands\\": Array of command objects to execute\\n\\nOptional fields:\\n- \\"task_complete\\": Boolean indicating if the task is complete (defaults to false if not present)\\n\\nCommand object structure:\\n- \\"keystrokes\\": String containing the exact keystrokes to send to the terminal (required)\\n- \\"duration\\": Number of seconds to wait for the command to complete before the next command will be executed (defaults to 1.0 if not present)\\n\\nIMPORTANT: The text inside \\"keystrokes\\" will be used completely verbatim as keystrokes. Write commands exactly as you want them sent to the terminal:\\n- Most bash commands should end with a newline (\\\\n) to cause them to execute\\n- For special key sequences, use tmux-style escape sequences:\\n  - C-c for Ctrl+C\\n  - C-d for Ctrl+D\\n\\nThe \\"duration\\" attribute specifies the number of seconds to wait for the command to complete (default: 1.0) before the next command will be executed. On immediate tasks (e.g., cd, ls, echo, cat) set a duration of 0.1 seconds. On commands (e.g., gcc, find, rustc) set a duration of 1.0 seconds. On slow commands (e.g., make, python3 [long running script], wget [file]) set an appropriate duration as you determine necessary.\\n\\nIt is better to set a smaller duration than a longer duration. It is always possible to wait again if the prior output has not finished, by running {\\"keystrokes\\": \\"\\", \\"duration\\": 10.0} on subsequent requests to wait longer. Never wait longer than 60 seconds; prefer to poll to see intermediate result status.\\n\\nImportant notes:\\n- Each command\'s keystrokes are sent exactly as written to the terminal\\n- Do not include extra whitespace before or after the keystrokes unless it\'s part of the intended command\\n- Extra text before or after the JSON will generate warnings but be tolerated\\n- The JSON must be valid - use proper escaping for quotes and special characters within strings\\n- Commands array can be empty if you want to wait without taking action\\n\\nTask Description:\\n# Student Grade Report Generator\\n\\nYou are a teaching assistant tasked with creating a grade summary report from raw student data.\\n\\n## Your Task\\n\\nProcess the input file `/app/student_grades.csv` and generate a JSON report at `/app/grade_summary.json` with the following requirements:\\n\\n1. **Read and parse** the CSV file which contains student grade data\\n2. **Filter** to include only students with a final grade of 70 or higher\\n3. **Group** the students by their course section\\n4. **Sort** the students within each section by their final grade in descending order (highest grade first)\\n5. **Transform** each student record to include only:\\n   - `student_id` (renamed from \\"id\\")\\n   - `name` (keep as is)\\n   - `final_grade` (renamed from \\"grade\\")\\n   - `passed` (set to `true` for all filtered students)\\n\\n## Input Format\\n\\nThe CSV file has the following columns:\\n- `id`: Student ID (string)\\n- `name`: Student name (string)\\n- `section`: Course section (string, e.g., \\"A\\", \\"B\\", \\"C\\")\\n- `grade`: Final numeric grade (integer 0-100)\\n- `attendance`: Attendance percentage (float)\\n\\n## Expected Output Format\\n\\nThe output JSON file must have this exact structure:\\n```json\\n{\\n  \\"sections\\": {\\n    \\"section_name\\": [\\n      {\\n        \\"student_id\\": \\"string\\",\\n        \\"name\\": \\"string\\",\\n        \\"final_grade\\": number,\\n        \\"passed\\": true\\n      }\\n    ]\\n  },\\n  \\"summary\\": {\\n    \\"total_students\\": number,\\n    \\"total_passing\\": number,\\n    \\"average_grade\\": number\\n  }\\n}\\n```\\n\\n**Requirements:**\\n- The JSON must be properly formatted with 2-space indentation\\n- Sections should be sorted alphabetically (A, B, C, etc.)\\n- Within each section, students must be sorted by final grade (highest to lowest)\\n- The `average_grade` in summary should be calculated as the average of all passing students\' grades, rounded to 2 decimal places\\n- Only include students with grade ≥ 70\\n\\n## Example\\n\\nGiven input:\\n```\\nid,name,section,grade,attendance\\n001,Alice,A,85,95.5\\n002,Bob,B,65,88.0\\n003,Charlie,A,92,91.0\\n004,Diana,C,78,87.5\\n```\\n\\nExpected output (partial):\\n```json\\n{\\n  \\"sections\\": {\\n    \\"A\\": [\\n      {\\n        \\"student_id\\": \\"003\\",\\n        \\"name\\": \\"Charlie\\",\\n        \\"final_grade\\": 92,\\n        \\"passed\\": true\\n      },\\n      {\\n        \\"student_id\\": \\"001\\",\\n        \\"name\\": \\"Alice\\",\\n        \\"final_grade\\": 85,\\n        \\"passed\\": true\\n      }\\n    ],\\n    \\"C\\": [\\n      {\\n        \\"student_id\\": \\"004\\",\\n        \\"name\\": \\"Diana\\",\\n        \\"final_grade\\": 78,\\n        \\"passed\\": true\\n      }\\n    ]\\n  },\\n  \\"summary\\": {\\n    \\"total_students\\": 4,\\n    \\"total_passing\\": 3,\\n    \\"average_grade\\": 85.0\\n  }\\n}\\n```\\n\\n## Verification\\n\\nYour solution will be tested by:\\n1. Checking that `/app/grade_summary.json` exists\\n2. Validating the JSON structure matches the expected format\\n3. Confirming the data is correctly filtered, grouped, and sorted\\n4. Verifying the summary statistics are accurate\\n\\nCurrent terminal state:\\nCurrent Terminal Screen:\\nroot@cpu1-0004:/app#\\n", "type": "text"}], "role": "user"}]'
EXPECTED_ASSISTANT_JSON = '{"content": [{"text": "We are in the /app directory. Need to process student_grades.csv and produce grade_summary.json. First, check if the input file exists and examine its structure. Then write a script (likely Python) to parse, filter, group, sort, and compute statistics. Finally, output JSON with proper formatting.", "type": "text"}], "reasoning_content": "We are in the /app directory. We need to process the CSV file and output a JSON file.\\n Let\'s first examine the input file to understand its structure.\\n Then we can write a script (likely in Python) to process the data and output the required JSON.\\n We\'ll need to install Python if not present, but likely it\'s installed. Let\'s check.\\n We\'ll also need to ensure we have the necessary libraries (csv, json, math). Standard library should suffice.\\n\\n Steps:\\n 1. Read the CSV file.\\n 2. Filter rows where grade >= 70.\\n 3. Group by section.\\n 4. Sort sections alphabetically.\\n 5. Within each section, sort students by grade descending.\\n 6. Transform each student to the required format.\\n 7. Calculate summary statistics:\\n    - total_students: total number of students in the CSV (including failing ones? According to example: total_students is 4, which is all students in the input, not just passing).\\n        Wait: The example says total_students: 4. The input has 4 students, and 3 passing. So total_students is the total number of students in the input (all students, regardless of passing).\\n    - total_passing: number of students with grade >= 70.\\n    - average_grade: average of the passing students\' grades (rounded to 2 decimal places).\\n\\n 8. Write the JSON with 2-space indentation.\\n\\n Let\'s check the current directory and the CSV file.", "role": "assistant", "tool_calls": [{"function": {"arguments": "{\\"command\\": \\"ls -la\\", \\"security_risk\\": \\"LOW\\", \\"summary\\": \\"We are in the /app directory. Need to process student_grades.csv and produce grade_summary.json. First, check if the inp\\"}", "name": "terminal"}, "id": "call_000001_terminal", "type": "function"}]}'
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
