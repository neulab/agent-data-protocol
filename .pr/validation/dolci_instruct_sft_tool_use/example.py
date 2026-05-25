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


DATASET_NAME = 'dolci_instruct_sft_tool_use'
MODEL = os.getenv("LLM_MODEL", "openhands/minimax-m2.7")
TARGET_TOOL = 'weather_forecast_weather_api'
TARGET_ARGUMENTS = '{\n  "days": 5,\n  "q": "Paris",\n  "security_risk": "UNKNOWN",\n  "summary": "Call weather_forecast_weather_api"\n}'
USED_TOOL_NAMES = '[\n  "cell_density",\n  "combinatorics_permutation_count",\n  "finish",\n  "get_all_predictions",\n  "get_city_from_zipcode",\n  "get_matches_on_a_specific_date",\n  "is_power_of_two",\n  "laliga_standings",\n  "leaguepowerrankingrounds",\n  "match_details_by_id",\n  "physics_final_velocity",\n  "reserve_hotel_room",\n  "schools",\n  "select_race_based_on_race_number",\n  "weather_forecast_weather_api"\n]'
TOOL_SPECS_JSON = '[{"function": {"description": "Provide a placeholder for the `cell_density` tool in the committed Dolci sample.", "name": "cell_density", "parameters": {"properties": {"dilution": {"type": "string"}, "od": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": [], "type": "object"}}, "type": "function"}, {"function": {"description": "Provide a placeholder for the `combinatorics_permutation_count` tool in the committed Dolci sample.", "name": "combinatorics_permutation_count", "parameters": {"properties": {"k": {"type": "string"}, "n": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": [], "type": "object"}}, "type": "function"}, {"function": {"description": "Signals completion of the current task or conversation.", "name": "finish", "parameters": {"properties": {"message": {"description": "Final message to send to the user.", "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": ["message"], "type": "object"}}, "type": "function"}, {"function": {"description": "Provide a placeholder for the `get_all_predictions` tool in the committed Dolci sample.", "name": "get_all_predictions", "parameters": {"properties": {"security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "sort": {"type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": [], "type": "object"}}, "type": "function"}, {"function": {"description": "Provide a placeholder for the `get_city_from_zipcode` tool in the committed Dolci sample.", "name": "get_city_from_zipcode", "parameters": {"properties": {"security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "zipcode": {"type": "string"}}, "required": [], "type": "object"}}, "type": "function"}, {"function": {"description": "Provide a placeholder for the `get_matches_on_a_specific_date` tool in the committed Dolci sample.", "name": "get_matches_on_a_specific_date", "parameters": {"properties": {"date": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "utc_offset": {"type": "string"}}, "required": [], "type": "object"}}, "type": "function"}, {"function": {"description": "Provide a placeholder for the `is_power_of_two` tool in the committed Dolci sample.", "name": "is_power_of_two", "parameters": {"properties": {"num": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": [], "type": "object"}}, "type": "function"}, {"function": {"description": "Provide a placeholder for the `laliga_standings` tool in the committed Dolci sample.", "name": "laliga_standings", "parameters": {"properties": {"season": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": [], "type": "object"}}, "type": "function"}, {"function": {"description": "Provide a placeholder for the `leaguepowerrankingrounds` tool in the committed Dolci sample.", "name": "leaguepowerrankingrounds", "parameters": {"properties": {"seasonid": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "tournamentid": {"type": "string"}}, "required": [], "type": "object"}}, "type": "function"}, {"function": {"description": "Provide a placeholder for the `match_details_by_id` tool in the committed Dolci sample.", "name": "match_details_by_id", "parameters": {"properties": {"match_id": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": [], "type": "object"}}, "type": "function"}, {"function": {"description": "Provide a placeholder for the `physics_final_velocity` tool in the committed Dolci sample.", "name": "physics_final_velocity", "parameters": {"properties": {"acceleration": {"type": "string"}, "initial_velocity": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}, "time": {"type": "string"}}, "required": [], "type": "object"}}, "type": "function"}, {"function": {"description": "Provide a placeholder for the `reserve_hotel_room` tool in the committed Dolci sample.", "name": "reserve_hotel_room", "parameters": {"properties": {"checkin_date": {"type": "string"}, "checkout_date": {"type": "string"}, "guest_id": {"type": "string"}, "nightly_rate": {"type": "string"}, "room_type": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": [], "type": "object"}}, "type": "function"}, {"function": {"description": "Provide a placeholder for the `schools` tool in the committed Dolci sample.", "name": "schools", "parameters": {"properties": {"identifier": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": [], "type": "object"}}, "type": "function"}, {"function": {"description": "Provide a placeholder for the `select_race_based_on_race_number` tool in the committed Dolci sample.", "name": "select_race_based_on_race_number", "parameters": {"properties": {"race_no": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": [], "type": "object"}}, "type": "function"}, {"function": {"description": "Provide a placeholder for the `weather_forecast_weather_api` tool in the committed Dolci sample.", "name": "weather_forecast_weather_api", "parameters": {"properties": {"days": {"type": "string"}, "q": {"type": "string"}, "security_risk": {"description": "Security risk for the action.", "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"], "type": "string"}, "summary": {"description": "Concise summary of what this action does.", "type": "string"}}, "required": [], "type": "object"}}, "type": "function"}]'


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
