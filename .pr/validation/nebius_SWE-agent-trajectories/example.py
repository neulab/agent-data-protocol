from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Self

from openhands.sdk import LLM, Message, TextContent
from openhands.sdk.tool import ToolDefinition
from openhands.sdk.tool.schema import Action


DATASET_NAME = 'nebius_SWE-agent-trajectories'
MODEL = os.getenv("LLM_MODEL", "openhands/minimax-m2.7")
TARGET_TOOL = 'terminal'
TARGET_ARGUMENTS = '{\n  "command": "ls -F",\n  "security_risk": "LOW",\n  "summary": "To start addressing the issue, we need to locate the `image.py` file and examine the line that is causing the `SyntaxErr"\n}'
USED_TOOL_NAMES = '[\n  "browser_navigate",\n  "create",\n  "edit",\n  "find_file",\n  "finish",\n  "open",\n  "search_file",\n  "terminal"\n]'
TOOL_SPECS_JSON = '[{"function": {"description": "Navigate to a URL in the browser.", "name": "browser_navigate", "parameters": {"properties": {"new_tab": {"description": "Whether to open in a new tab.", "type": "boolean"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "url": {"description": "URL to navigate to.", "type": "string"}}, "required": ["url"], "type": "object"}}, "type": "function"}, {"function": {"description": "Create and open a new file with the given name.\\n\\nArgs:\\n----\\n    filename (str): The name of the file to create.", "name": "create", "parameters": {"properties": {"filename": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["filename"], "type": "object"}}, "type": "function"}, {"function": {"description": "Replace lines from start_line to end_line (inclusive) with the given text in the open file. The replacement text is terminated by a line with only \'end_of_edit\' on it. All of the replacement text will be entered, so make sure your indentation is formatted properly. Python files will be checked for syntax errors after the edit. If the system detects a syntax error, the edit will not be executed. Simply try to edit the file again, but make sure to read the error message and modify the edit command you issue accordingly. Issuing the same command a second time will just lead to the same error message again. Please note that THE EDIT COMMAND REQUIRES PROPER INDENTATION. If you\'d like to add the line \'        pr int(x)\' you must fully write that out, with all those spaces before the code! Indentation is important and code that is not indented correctly will fail and require fixing before it can be run.\\n\\nArgs:\\n----\\n    start_line (int): The line number to start the edit at.\\n    end_line (int): The line number to end the edit at (inclusive).\\n    replacement_text (str): The text to replace the current selection with.", "name": "edit", "parameters": {"properties": {"end_line": {"type": "string"}, "replacement_text": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "start_line": {"type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["start_line", "end_line", "replacement_text"], "type": "object"}}, "type": "function"}, {"function": {"description": "Find all files with the given name in the specified directory. If dir is not provided, search in the current directory.\\n\\nArgs:\\n----\\n    file_name (str): The name of the file to search for.\\n    dir (str, optional): The directory to search in (if not provided, search in the current directory).", "name": "find_file", "parameters": {"properties": {"dir": {"type": "string"}, "file_name": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["file_name"], "type": "object"}}, "type": "function"}, {"function": {"description": "Signals completion of the current task or conversation.", "name": "finish", "parameters": {"properties": {"message": {"description": "Final message to send to the user.", "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["message"], "type": "object"}}, "type": "function"}, {"function": {"description": "Open the file at the given path in the editor. If line_number is provided, the window will be moved to include that line.\\n\\nArgs:\\n----\\n    path (str): The path to the file to open.\\n    line_number (int, optional): The line number to move the window to (if not provided, the window will start at the top of the file).", "name": "open", "parameters": {"properties": {"line_number": {"type": "string"}, "path": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["path"], "type": "object"}}, "type": "function"}, {"function": {"description": "Search for the search_term in the specified file. If file is not provided, search in the current open file.\\n\\nArgs:\\n----\\n    search_term (str): The term to search for.\\n    file (str, optional): The file to search in (if not provided, search in the current open file).", "name": "search_file", "parameters": {"properties": {"file": {"type": "string"}, "search_term": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["search_term"], "type": "object"}}, "type": "function"}, {"function": {"description": "Execute a shell command in the terminal within a persistent shell session.", "name": "terminal", "parameters": {"properties": {"command": {"description": "The shell command or terminal input.", "type": "string"}, "is_input": {"description": "Whether command should be sent as input to a running process.", "type": "boolean"}, "reset": {"description": "Whether to reset the terminal session.", "type": "boolean"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "timeout": {"description": "Optional timeout in seconds.", "type": "number"}}, "required": ["command"], "type": "object"}}, "type": "function"}]'


class ValidationTool(ToolDefinition):
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


def make_tool(spec: dict) -> ToolDefinition:
    function = spec["function"]
    name = function["name"]
    parameters = function.get("parameters") or {"type": "object", "properties": {}}
    action_type = Action.from_mcp_schema(f"{class_name(name)}Action", parameters)
    tool_cls = type(f"{class_name(name)}Tool", (ValidationTool,), {"name": name})
    return tool_cls(
        description=function.get("description") or f"Validation tool {name}.",
        action_type=action_type,
        observation_type=None,
    )


def latest_log(log_dir: Path) -> Path:
    logs = sorted(log_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
    if not logs:
        raise RuntimeError("SDK did not write a completion log")
    return logs[-1]


def has_tool_call(log_path: Path) -> bool:
    data = json.loads(log_path.read_text())
    for choice in data.get("response", {}).get("choices", []):
        message = choice.get("message") or {}
        if message.get("tool_calls"):
            return True
    return False


def main() -> None:
    root = repo_root()
    load_env_file(root / ".env")
    load_env_file(Path.home() / "work" / "agent-data-protocol" / ".env")
    if not os.getenv("LLM_API_KEY"):
        raise RuntimeError("LLM_API_KEY is required")

    system_prompt = (root / "agents" / "openhands_sdk" / "system_prompt.txt").read_text()
    tools = [make_tool(spec) for spec in json.loads(TOOL_SPECS_JSON)]
    log_dir = Path(tempfile.mkdtemp(prefix=f"{DATASET_NAME}-completion-"))
    llm = LLM(
        model=MODEL,
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        log_completions=True,
        log_completions_folder=str(log_dir),
        max_output_tokens=160,
    )
    prompt = (
        f"This is an OpenHands SDK logging validation for the ADP dataset "
        f"{DATASET_NAME}. The available tools are: {', '.join(USED_TOOL_NAMES) or TARGET_TOOL}. "
        f"Call exactly one tool now: `{TARGET_TOOL}`. Use arguments similar to this JSON: "
        f"{TARGET_ARGUMENTS}. Do not answer in plain text only."
    )
    last_log = None
    for _ in range(2):
        llm.completion(
            messages=[
                Message(role="system", content=[TextContent(text=system_prompt)]),
                Message(role="user", content=[TextContent(text=prompt)]),
            ],
            tools=tools,
            add_security_risk_prediction=True,
            temperature=0,
            tool_choice={"type": "function", "function": {"name": TARGET_TOOL}},
        )
        last_log = latest_log(log_dir)
        if has_tool_call(last_log):
            break
        prompt = (
            f"You must call the `{TARGET_TOOL}` tool exactly once for validation. "
            f"Use arguments like: {TARGET_ARGUMENTS}."
        )
    assert last_log is not None
    shutil.copyfile(last_log, Path(__file__).with_name("completion.json"))


if __name__ == "__main__":
    main()
