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


DATASET_NAME = 'toucan_1_5m'
MODEL = os.getenv("LLM_MODEL", "openhands/minimax-m2.7")
TARGET_TOOL = 'web_search_exa'
TARGET_ARGUMENTS = '{\n  "numResults": 5,\n  "query": "StellarPay fintech startup Series B business model recent developments",\n  "security_risk": "UNKNOWN",\n  "summary": "I\'ll help you conduct comprehensive research on StellarPay to inform your decision. Let me search for information about "\n}'
USED_TOOL_NAMES = '[\n  "company_research_exa",\n  "finish",\n  "linkedin_search_exa",\n  "web_search_exa"\n]'
TOOL_SPECS_JSON = '[{"function": {"description": "Research companies using Exa AI - finds comprehensive information about businesses, organizations, and corporations.\\n\\nArgs:\\n----\\n    companyName: Name of the company to research\\n    numResults: Number of search results to return (default: 5)\\n\\nReturns:\\n-------\\n    dict: Response from the API", "name": "company_research_exa", "parameters": {"properties": {"companyName": {"type": "string"}, "numResults": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["companyName"], "type": "object"}}, "type": "function"}, {"function": {"description": "Signals completion of the current task or conversation.", "name": "finish", "parameters": {"properties": {"message": {"description": "Final message to send to the user.", "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["message"], "type": "object"}}, "type": "function"}, {"function": {"description": "Search LinkedIn profiles and companies using Exa AI - finds professional profiles, company pages, and business-related content on LinkedIn.\\n\\nArgs:\\n----\\n    query: LinkedIn search query (e.g., person name, company, job title)\\n    searchType: Type of LinkedIn content to search (default: all)\\n    numResults: Number of LinkedIn results to return (default: 5)\\n\\nReturns:\\n-------\\n    dict: Response from the API", "name": "linkedin_search_exa", "parameters": {"properties": {"numResults": {"type": "string"}, "query": {"type": "string"}, "searchType": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["query"], "type": "object"}}, "type": "function"}, {"function": {"description": "Search the web using Exa AI - performs real-time web searches and can scrape content from specific URLs.\\n\\nArgs:\\n----\\n    query: Search query\\n    numResults: Number of search results to return (default: 5)\\n\\nReturns:\\n-------\\n    dict: Response from the API", "name": "web_search_exa", "parameters": {"properties": {"numResults": {"type": "string"}, "query": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["query"], "type": "object"}}, "type": "function"}]'


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
