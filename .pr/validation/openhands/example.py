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


DATASET_NAME = 'openhands'
MODEL = os.getenv("LLM_MODEL", "openhands/minimax-m2.7")
TARGET_TOOL = 'terminal'
TARGET_ARGUMENTS = '{\n  "command": "python - <<\'PY\'\\nopen_file(\'/workspace/openhands/runtime/client/client.py\')\\nPY",\n  "security_risk": "MEDIUM",\n  "summary": "Certainly! I\'ll add an endpoint to the FastAPI server in the `client.py` file to provide information about the server\'s "\n}'
USED_TOOL_NAMES = '[\n  "browser_click",\n  "browser_navigate",\n  "browser_scroll",\n  "crawl",\n  "delegate_to_CrawlAgent",\n  "delegate_to_RagAgent",\n  "delegate_to_agent",\n  "finish",\n  "rag_search",\n  "send_msg_to_user",\n  "terminal"\n]'
TOOL_SPECS_JSON = '[{"function": {"description": "Click an element on the page by its current browser state index.", "name": "browser_click", "parameters": {"properties": {"index": {"description": "Index from the browser state output.", "type": "integer"}, "new_tab": {"description": "Whether to open navigation in a new tab.", "type": "boolean"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["index"], "type": "object"}}, "type": "function"}, {"function": {"description": "Navigate to a URL in the browser.", "name": "browser_navigate", "parameters": {"properties": {"new_tab": {"description": "Whether to open in a new tab.", "type": "boolean"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "url": {"description": "URL to navigate to.", "type": "string"}}, "required": ["url"], "type": "object"}}, "type": "function"}, {"function": {"description": "Scroll the page up or down.", "name": "browser_scroll", "parameters": {"properties": {"direction": {"enum": ["up", "down"], "type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["direction"], "type": "object"}}, "type": "function"}, {"function": {"description": "Crawl a webpage.\\n\\nArgs:\\n----\\n    link (str): The link to crawl.", "name": "crawl", "parameters": {"properties": {"link": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["link"], "type": "object"}}, "type": "function"}, {"function": {"description": "Delegate the task to the CrawlAgent.\\n\\nArgs:\\n----\\n    task (str): task description.\\n    link (str): the link to crawl.", "name": "delegate_to_CrawlAgent", "parameters": {"properties": {"link": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "task": {"type": "string"}}, "required": ["task", "link"], "type": "object"}}, "type": "function"}, {"function": {"description": "Delegate the task to the RagAgent.\\n\\nArgs:\\n----\\n    task (str): task description.\\n    query (str): the query to search.", "name": "delegate_to_RagAgent", "parameters": {"properties": {"query": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "task": {"type": "string"}}, "required": ["task", "query"], "type": "object"}}, "type": "function"}, {"function": {"description": "Delegate the task to the BrowsingAgent.\\n\\nArgs:\\n----\\n    agent (str): The agent to delegate the task to.\\n    task (str): task description.", "name": "delegate_to_agent", "parameters": {"properties": {"agent": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "task": {"type": "string"}}, "required": ["agent", "task"], "type": "object"}}, "type": "function"}, {"function": {"description": "Signals completion of the current task or conversation.", "name": "finish", "parameters": {"properties": {"message": {"description": "Final message to send to the user.", "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["message"], "type": "object"}}, "type": "function"}, {"function": {"description": "Search using the RAG model.\\n\\nArgs:\\n----\\n    query (str): The query to search.", "name": "rag_search", "parameters": {"properties": {"query": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["query"], "type": "object"}}, "type": "function"}, {"function": {"description": "Send a message to the user.\\n\\nArgs:\\n----\\n    msg (str): The message content to send.\\n\\nExample: send_msg_to_user(\'The number of stars for the React repository on GitHub is 225k.\')", "name": "send_msg_to_user", "parameters": {"properties": {"msg": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["msg"], "type": "object"}}, "type": "function"}, {"function": {"description": "Execute a shell command in the terminal within a persistent shell session.", "name": "terminal", "parameters": {"properties": {"command": {"description": "The shell command or terminal input.", "type": "string"}, "is_input": {"description": "Whether command should be sent as input to a running process.", "type": "boolean"}, "reset": {"description": "Whether to reset the terminal session.", "type": "boolean"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "timeout": {"description": "Optional timeout in seconds.", "type": "number"}}, "required": ["command"], "type": "object"}}, "type": "function"}]'


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
