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


DATASET_NAME = 'miroverse_v0_1'
MODEL = os.getenv("LLM_MODEL", "openhands/minimax-m2.7")
TARGET_TOOL = 'browsing_agent__search_and_browse'
TARGET_ARGUMENTS = '{\n  "security_risk": "UNKNOWN",\n  "subtask": "Find the date of the boxing match between Ran Nakash and Marco Huck.",\n  "summary": "To find the date of Ran Nakash\'s only loss against Marco Huck, I will follow these steps: 1. Search for information abou"\n}'
USED_TOOL_NAMES = '[\n  "browsing_agent__search_and_browse",\n  "finish",\n  "tool_google_search__google_search",\n  "tool_google_search__scrape",\n  "tool_serper_search__google_search",\n  "tool_serper_search__scrape"\n]'
TOOL_SPECS_JSON = '[{"function": {"description": "[browsing-agent/search_and_browse] This tool is an agent that performs the subtask of searching and browsing the web for specific missing information and generating the desired answer. The subtask should be clearly defined, include relevant background, and focus on factual gaps. It does not perform vague or speculative subtasks.\\n\\nArgs:\\n        subtask: the subtask to be performed.\\n\\nReturns:\\n        the result of the subtask.", "name": "browsing_agent__search_and_browse", "parameters": {"properties": {"security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "subtask": {"type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["subtask"], "type": "object"}}, "type": "function"}, {"function": {"description": "Signals completion of the current task or conversation.", "name": "finish", "parameters": {"properties": {"message": {"description": "Final message to send to the user.", "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["message"], "type": "object"}}, "type": "function"}, {"function": {"description": "[tool-google-search/google_search] Tool to perform web searches via Serper API and retrieve rich results. It is able to retrieve organic search results, people also ask, related searches, and knowledge graph.\\n\\nArgs:\\n    q: Search query string\\n    gl: Optional region code for search results in ISO 3166-1 alpha-2 format (e.g., \'us\')\\n    hl: Optional language code for search results in ISO 639-1 format (e.g., \'en\')\\n    location: Optional location for search results (e.g., \'SoHo, New York, United States\', \'California, United States\')\\n    num: Number of results to return (default: 10)\\n    tbs: Time-based search filter (\'qdr:h\' for past hour, \'qdr:d\' for past day, \'qdr:w\' for past week, \'qdr:m\' for past month, \'qdr:y\' for past year)\\n    page: Page number of results to return (default: 1)\\n    autocorrect: Whether to autocorrect spelling in query", "name": "tool_google_search__google_search", "parameters": {"properties": {"autocorrect": {"type": "string"}, "gl": {"type": "string"}, "hl": {"type": "string"}, "location": {"type": "string"}, "num": {"type": "string"}, "page": {"type": "string"}, "q": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "tbs": {"type": "string"}}, "required": ["q"], "type": "object"}}, "type": "function"}, {"function": {"description": "[tool-google-search/scrape] Tool to scrape a webpage and retrieve the text and, optionally, the markdown content. It will retrieve also the JSON-LD metadata and the head metadata.\\n\\nArgs:\\n    url: The URL of the webpage to scrape.\\n    includeMarkdown: Whether to include markdown content.", "name": "tool_google_search__scrape", "parameters": {"properties": {"includeMarkdown": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "url": {"type": "string"}}, "required": ["url"], "type": "object"}}, "type": "function"}, {"function": {"description": "[tool-serper-search/google_search] Tool to perform web searches via Serper API and retrieve rich results. It is able to retrieve organic search results, people also ask, related searches, and knowledge graph.\\n\\nArgs:\\n    q: Search query string\\n    gl: Optional region code for search results in ISO 3166-1 alpha-2 format (e.g., \'us\')\\n    hl: Optional language code for search results in ISO 639-1 format (e.g., \'en\')\\n    location: Optional location for search results (e.g., \'SoHo, New York, United States\', \'California, United States\')\\n    num: Number of results to return (default: 10)\\n    tbs: Time-based search filter (\'qdr:h\' for past hour, \'qdr:d\' for past day, \'qdr:w\' for past week, \'qdr:m\' for past month, \'qdr:y\' for past year)\\n    page: Page number of results to return (default: 1)\\n    autocorrect: Whether to autocorrect spelling in query", "name": "tool_serper_search__google_search", "parameters": {"properties": {"autocorrect": {"type": "string"}, "gl": {"type": "string"}, "hl": {"type": "string"}, "location": {"type": "string"}, "num": {"type": "string"}, "page": {"type": "string"}, "q": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "tbs": {"type": "string"}}, "required": ["q"], "type": "object"}}, "type": "function"}, {"function": {"description": "[tool-serper-search/scrape] Tool to scrape a webpage and retrieve the text and, optionally, the markdown content. It will retrieve also the JSON-LD metadata and the head metadata.\\n\\nArgs:\\n    url: The URL of the webpage to scrape.\\n    includeMarkdown: Whether to include markdown content.", "name": "tool_serper_search__scrape", "parameters": {"properties": {"includeMarkdown": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "url": {"type": "string"}}, "required": ["url"], "type": "object"}}, "type": "function"}]'


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
