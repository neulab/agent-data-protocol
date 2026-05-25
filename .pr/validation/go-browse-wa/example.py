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


DATASET_NAME = 'go-browse-wa'
MODEL = os.getenv("LLM_MODEL", "openhands/minimax-m2.7")
RECORD_ID = '0'
ASSISTANT_INDEX = 3
PREFIX_MESSAGES_JSON = '[{"content": [{"text": "You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.\\n\\n<ROLE>\\n* Your primary role is to assist users by executing commands, modifying code, and solving technical problems effectively. You should be thorough, methodical, and prioritize quality over speed.\\n* If the user asks a question, like \\"why is X happening\\", don\'t try to fix the problem. Just give an answer to the question.\\n</ROLE>\\n\\n<MEMORY>\\n* Use `AGENTS.md` under the repository root as your persistent memory for repository-specific knowledge and context.\\n* Add important insights, patterns, and learnings to this file to improve future task performance.\\n* This repository skill is automatically loaded for every conversation and helps maintain context across sessions.\\n</MEMORY>\\n\\n<EFFICIENCY>\\n* Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action, e.g. combine multiple bash commands into one, using sed and grep to edit/view multiple files at once.\\n* When exploring the codebase, use efficient tools like find, grep, and git commands with appropriate filters to minimize unnecessary operations.\\n</EFFICIENCY>\\n\\n<FILE_SYSTEM_GUIDELINES>\\n* When a user provides a file path, do NOT assume it\'s relative to the current working directory. First explore the file system to locate the file before working on it.\\n* If asked to edit a file, edit the file directly, rather than creating a new file with a different filename.\\n* For global search-and-replace operations, consider using `sed` instead of opening file editors multiple times.\\n* NEVER create multiple versions of the same file with different suffixes. Modify the original file directly when making changes.\\n</FILE_SYSTEM_GUIDELINES>\\n\\n<CODE_QUALITY>\\n* Write clean, efficient code with minimal comments.\\n* Before implementing changes, first thoroughly understand the codebase through exploration.\\n* Place imports at the top of the file unless doing so would cause circular imports or other real issues.\\n</CODE_QUALITY>\\n\\n<VERSION_CONTROL>\\n* Exercise caution with git operations. Do NOT make potentially dangerous changes unless explicitly asked to do so.\\n* When committing changes, use `git status` to see all modified files, and stage all files necessary for the commit.\\n* Do NOT commit files that typically should not go into version control unless explicitly instructed by the user.\\n</VERSION_CONTROL>\\n\\n<PROBLEM_SOLVING_WORKFLOW>\\n1. EXPLORATION: Thoroughly explore relevant files and understand the context before proposing solutions.\\n2. ANALYSIS: Consider multiple approaches and select the most promising one.\\n3. TESTING: For bug fixes, create tests to verify issues before implementing fixes when appropriate.\\n4. IMPLEMENTATION: Make focused, minimal changes to address the problem.\\n5. VERIFICATION: Test your implementation thoroughly, including edge cases.\\n</PROBLEM_SOLVING_WORKFLOW>\\n\\n<SECURITY_RISK_ASSESSMENT>\\nWhen using tools that support the security_risk parameter, assess the safety risk of your actions:\\n* LOW: Safe, read-only actions.\\n* MEDIUM: Project-scoped edits or execution.\\n* HIGH: System-level or untrusted operations, privilege changes, or sensitive-data movement.\\n</SECURITY_RISK_ASSESSMENT>\\n\\n<EXTERNAL_SERVICES>\\n* When interacting with external services like GitHub, GitLab, or Bitbucket, use their respective APIs instead of browser-based interactions whenever possible.\\n* Only resort to browser-based interactions with these services if specifically requested by the user or if the required operation cannot be performed via API.\\n</EXTERNAL_SERVICES>\\n\\n<ENVIRONMENT_SETUP>\\n* When user asks you to run an application, do not stop if the application is not installed. Instead, install the application and run the command again when possible.\\n* If you encounter missing dependencies, first look around in the repository for existing dependency files and use them to install dependencies.\\n</ENVIRONMENT_SETUP>\\n", "type": "text"}], "role": "system"}, {"content": [{"text": "Find driving directions from Central Park, New York to Times Square, New York using the Car (OSRM) option.", "type": "text"}], "role": "user"}, {"content": [{"text": "RootWebArea \'OpenStreetMap\', focused, url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/#map=7/42.896/-75.108\'\\n\\t[44] banner \'\'\\n\\t\\t[45] heading \'OpenStreetMap logo OpenStreetMap\'\\n\\t\\t\\t[46] link \'OpenStreetMap logo OpenStreetMap\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/#map=7/42.896/-75.108\'\\n\\t\\t\\t\\t[49] image \'OpenStreetMap logo\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/assets/osm_logo-4afddaae0230a5a46687fdc751ed256dfdccde144118cb02a7d7960f207a4b92.svg\'\\n\\t\\t\\t\\tStaticText \'\'\\n\\t\\t[51] navigation \'\'\\n\\t\\t\\t[95] link \'Edit\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/edit#map=7/42.896/-75.108\'\\n\\t\\t\\t[96] button \'\'\\n\\t\\t\\t[102] link \'History\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/history#map=7/42.896/-75.108\'\\n\\t\\t\\t[103] link \'Export\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/export#map=7/42.896/-75.108\'\\n\\t\\t[104] navigation \'\'\\n\\t\\t\\t[105] list \'\'\\n\\t\\t\\t\\t[106] listitem \'\'\\n\\t\\t\\t\\t\\t[107] link \'GPS Traces\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/traces\'\\n\\t\\t\\t\\t[108] listitem \'\'\\n\\t\\t\\t\\t\\t[109] link \'User Diaries\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/diary\'\\n\\t\\t\\t\\t[110] listitem \'\'\\n\\t\\t\\t\\t\\t[111] link \'Communities\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/communities\'\\n\\t\\t\\t\\t[112] listitem \'\'\\n\\t\\t\\t\\t\\t[113] link \'Copyright\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/copyright\'\\n\\t\\t\\t\\t[114] listitem \'\'\\n\\t\\t\\t\\t\\t[115] link \'Help\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/help\'\\n\\t\\t\\t\\t[116] listitem \'\'\\n\\t\\t\\t\\t\\t[117] link \'About\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/about\'\\n\\t\\t\\t[134] link \'Log In\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/login?referer=%2F#map=7/42.896/-75.108\'\\n\\t\\t\\t[135] link \'Sign Up\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/user/new\'\\n\\t[144] link \'Where is this?\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/#\'\\n\\t[145] textbox \'Search\', focused\\n\\t[147] button \'Go\'\\n\\t[149] link \'Find directions between two points\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/directions\'\\n\\t\\t[150] image \'\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/assets/directions-b08069e78ff09f082d27a7434b49685efa647060cf51a71d74074ad350e3fd66.png\'\\n\\t[187] heading \'Welcome to OpenStreetMap!\'\\n\\t[189] button \'Close\'\\n\\t[191] paragraph \'\'\\n\\t\\tStaticText \'OpenStreetMap is a map of the world, created by people like you and free to use under an open license.\'\\n\\t[192] paragraph \'\'\\n\\t\\tStaticText \'Hosting is supported by\'\\n\\t\\t[193] link \'UCL\', url=\'https://www.ucl.ac.uk/\'\\n\\t\\tStaticText \',\'\\n\\t\\t[194] link \'Fastly\', url=\'https://www.fastly.com/\'\\n\\t\\tStaticText \',\'\\n\\t\\t[195] link \'Bytemark Hosting\', url=\'https://www.bytemark.co.uk/\'\\n\\t\\tStaticText \', and other\'\\n\\t\\t[196] link \'partners\', url=\'https://hardware.openstreetmap.org/thanks/\'\\n\\t\\tStaticText \'.\'\\n\\t[199] link \'Learn More\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/about\'\\n\\t[201] link \'Start Mapping\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/user/new\'\\n\\t[360] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/36/46.png\'\\n\\t[361] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/37/46.png\'\\n\\t[362] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/36/47.png\'\\n\\t[363] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/37/47.png\'\\n\\t[364] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/36/45.png\'\\n\\t[365] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/37/45.png\'\\n\\t[366] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/35/46.png\'\\n\\t[367] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/38/46.png\'\\n\\t[368] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/35/47.png\'\\n\\t[369] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/38/47.png\'\\n\\t[370] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/36/48.png\'\\n\\t[371] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/37/48.png\'\\n\\t[372] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/35/45.png\'\\n\\t[373] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/38/45.png\'\\n\\t[374] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/35/48.png\'\\n\\t[375] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/38/48.png\'\\n\\t[376] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/36/44.png\'\\n\\t[377] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/37/44.png\'\\n\\t[378] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/34/46.png\'\\n\\t[379] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/39/46.png\'\\n\\t[380] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/34/47.png\'\\n\\t[381] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/39/47.png\'\\n\\t[382] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/36/49.png\'\\n\\t[383] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/37/49.png\'\\n\\t[384] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/35/44.png\'\\n\\t[385] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/38/44.png\'\\n\\t[386] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/34/45.png\'\\n\\t[387] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/39/45.png\'\\n\\t[388] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/34/48.png\'\\n\\t[389] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/39/48.png\'\\n\\t[390] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/35/49.png\'\\n\\t[391] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/38/49.png\'\\n\\t[392] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/34/44.png\'\\n\\t[393] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/39/44.png\'\\n\\t[394] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/34/49.png\'\\n\\t[395] image \'\', url=\'http://ogma.lti.cs.cmu.edu:8080/tile/7/39/49.png\'\\n\\t[406] link \'Zoom In\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/#\'\\n\\t[408] link \'Zoom Out\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/#\'\\n\\t[411] button \'Show My Location\'\\n\\t[414] link \'Layers\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/#\'\\n\\t[417] link \'\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/#\'\\n\\t[420] link \'Share\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/#\'\\n\\t[423] link \'\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/#\'\\n\\t[426] link \'\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/#\'\\n\\tStaticText \'50 km\'\\n\\tStaticText \'50 mi\'\\n\\tStaticText \'©\'\\n\\t[434] link \'OpenStreetMap contributors\', url=\'http://ec2-3-148-123-246.us-east-2.compute.amazonaws.com:3000/copyright\'\\n\\tStaticText \'♥\'\\n\\t[435] link \'Make a Donation\', url=\'https://donate.openstreetmap.org/\'\\n\\tStaticText \'.\'\\n\\t[436] link \'Website and API terms\', url=\'https://wiki.osmfoundation.org/wiki/Terms_of_Use\'", "type": "text"}], "role": "user"}]'
EXPECTED_ASSISTANT_JSON = '{"content": [{"text": "Looking at the current page, I can see I\'m on OpenStreetMap. To find directions, I need to click on the \'Find directions between two points\' link which has a directions icon. I can see this element with ID 149 in the accessibility tree.", "type": "text"}], "role": "assistant", "tool_calls": [{"function": {"arguments": "{\\"index\\": 149, \\"security_risk\\": \\"MEDIUM\\", \\"summary\\": \\"Looking at the current page, I can see I\'m on OpenStreetMap. To find directions, I need to click on the \'Find directions\\"}", "name": "browser_click"}, "id": "call_000001_browser_click", "type": "function"}]}'
TOOL_SPECS_JSON = '[{"function": {"description": "Click an element on the page by its current browser state index.", "name": "browser_click", "parameters": {"properties": {"index": {"description": "Index from the browser state output.", "type": "integer"}, "new_tab": {"description": "Whether to open navigation in a new tab.", "type": "boolean"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["index"], "type": "object"}}, "type": "function"}, {"function": {"description": "Type text into an input element.", "name": "browser_type", "parameters": {"properties": {"index": {"description": "Index from the browser state output.", "type": "integer"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "text": {"description": "Text to type.", "type": "string"}}, "required": ["index", "text"], "type": "object"}}, "type": "function"}, {"function": {"description": "View, create, and edit files in plain-text format.", "name": "file_editor", "parameters": {"properties": {"command": {"enum": ["view", "create", "str_replace", "insert", "undo_edit"], "type": "string"}, "file_text": {"description": "Content for create operations.", "type": "string"}, "insert_line": {"description": "Line before which to insert.", "type": "integer"}, "new_str": {"description": "Replacement text.", "type": "string"}, "old_str": {"description": "Text to replace.", "type": "string"}, "path": {"description": "Absolute file path.", "type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "view_range": {"description": "Optional inclusive line range for view.", "items": {"type": "integer"}, "type": "array"}}, "required": ["command", "path"], "type": "object"}}, "type": "function"}, {"function": {"description": "Signals completion of the current task or conversation.", "name": "finish", "parameters": {"properties": {"message": {"description": "Final message to send to the user.", "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["message"], "type": "object"}}, "type": "function"}, {"function": {"description": "Create, update, and inspect the task list for the current conversation.", "name": "task_tracker", "parameters": {"properties": {"command": {"enum": ["plan", "update", "view"], "type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "task_list": {"description": "Markdown task list content.", "type": "string"}}, "required": [], "type": "object"}}, "type": "function"}, {"function": {"description": "Execute a shell command in the terminal within a persistent shell session.", "name": "terminal", "parameters": {"properties": {"command": {"description": "The shell command or terminal input.", "type": "string"}, "is_input": {"description": "Whether command should be sent as input to a running process.", "type": "boolean"}, "reset": {"description": "Whether to reset the terminal session.", "type": "boolean"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "timeout": {"description": "Optional timeout in seconds.", "type": "number"}}, "required": ["command"], "type": "object"}}, "type": "function"}, {"function": {"description": "Log a thought without obtaining new information or changing the environment.", "name": "think", "parameters": {"properties": {"summary": {"description": "Concise summary of what this action does.", "type": "string"}, "thought": {"description": "The thought to log.", "type": "string"}}, "required": ["thought"], "type": "object"}}, "type": "function"}]'


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
