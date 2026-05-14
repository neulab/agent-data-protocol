import ast
import json
import re
import sys
from typing import Any

from schema_raw import SchemaRaw

from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

ACTION_RE = re.compile(r"(?:^|\n)(?:Action|Act)\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
THOUGHT_RE = re.compile(
    r"^\s*(?:Thought|Think)\s*:\s*(.*?)(?:\n\s*(?:Action|Act)\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)
CODE_BLOCK_RE = re.compile(r"```(?:python|json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
FINAL_ANSWER_RE = re.compile(r"Final Answer\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
FUNCTION_STYLE_ACTION_RE = re.compile(r"^([A-Za-z_][\w.]*)\((.*)\)$", re.DOTALL)


def python_literal(value: Any) -> str:
    return repr(value)


def parse_structured_value(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text


def parse_json_tool_call(content: str) -> ApiAction | None:
    parsed = parse_structured_value(content)
    if not isinstance(parsed, dict):
        return None

    if "Tool" in parsed:
        tool = parsed["Tool"]
        parameters = parsed.get("Param", {})
        description = parsed.get("goal")
    elif "Action" in parsed:
        tool = parsed["Action"]
        parameters = parsed.get("Param", parsed.get("args", {}))
        description = parsed.get("Thought") or parsed.get("thought") or parsed.get("goal")
    elif "name" in parsed and "parameters" in parsed:
        tool = parsed["name"]
        parameters = parsed.get("parameters", {})
        description = None
    else:
        return None

    return ApiAction(
        function="call_api",
        kwargs={"tool": python_literal(tool), "parameters": python_literal(parameters)},
        description=description,
    )


def parse_code_block_tool_call(content: str) -> ApiAction | None:
    match = CODE_BLOCK_RE.search(content)
    if not match:
        return None

    parsed = parse_structured_value(match.group(1))
    if not isinstance(parsed, dict) or "name" not in parsed:
        return None

    description = content[: match.start()].strip() or None
    return ApiAction(
        function="call_api",
        kwargs={
            "tool": python_literal(parsed["name"]),
            "parameters": python_literal(parsed.get("parameters", {})),
        },
        description=description,
    )


def extract_thought(content: str) -> str | None:
    match = THOUGHT_RE.search(content)
    if not match:
        return None
    return match.group(1).strip() or None


def parse_action_line(content: str) -> ApiAction | MessageAction | None:
    match = ACTION_RE.search(content)
    if not match:
        return None

    action_text = match.group(1).strip()
    if not action_text:
        return None

    thought = extract_thought(content)
    final_answer = FINAL_ANSWER_RE.search(action_text)
    if final_answer:
        return MessageAction(
            content=f"<finish> {final_answer.group(1).strip()} </finish>", description=thought
        )

    if " with Action Input:" in action_text:
        tool, action_input = action_text.split(" with Action Input:", 1)
        tool = tool.strip()
        parameters = parse_structured_value(action_input)
        if tool.lower() in {"finish", "finalaction"}:
            return MessageAction(
                content=f"<finish> {json.dumps(parameters, ensure_ascii=False)} </finish>",
                description=thought,
            )
        return ApiAction(
            function="call_api",
            kwargs={"tool": python_literal(tool), "parameters": python_literal(parameters)},
            description=thought,
        )

    if action_text.lower().startswith(("finish", "answer", "finalaction")):
        return MessageAction(content=f"<finish> {action_text} </finish>", description=thought)

    function_style = FUNCTION_STYLE_ACTION_RE.match(action_text)
    if function_style:
        return ApiAction(
            function="call_api",
            kwargs={
                "tool": python_literal(function_style.group(1)),
                "parameters": python_literal({"arguments": function_style.group(2)}),
            },
            description=thought,
        )

    return ApiAction(
        function="perform_action",
        kwargs={"action": python_literal(action_text)},
        description=thought,
    )


def convert_assistant(content: str) -> ApiAction | MessageAction:
    normalized = content.strip().lower()
    if normalized in {"ok.", "ok", "okay."} or "i'll follow your instructions" in normalized:
        return MessageAction(content=content)

    for parser in (parse_json_tool_call, parse_code_block_tool_call, parse_action_line):
        converted = parser(content)
        if converted is not None:
            return converted

    return MessageAction(content=content)


def is_environment_observation(content: str, seen_action: bool) -> bool:
    stripped = content.strip()
    return seen_action or stripped.startswith("Observation:") or stripped.startswith('{"error"')


def process_data(raw_data: dict) -> Trajectory:
    data = SchemaRaw(**raw_data)
    content = []
    seen_action = False

    for turn in data.conversations:
        if turn.role == "assistant":
            converted = convert_assistant(turn.content)
            if isinstance(converted, ApiAction):
                seen_action = True
            content.append(converted)
        elif turn.role in {"user", "system"}:
            source = (
                "environment" if is_environment_observation(turn.content, seen_action) else "user"
            )
            content.append(TextObservation(content=turn.content, source=source))
        else:
            content.append(TextObservation(content=turn.content, source="environment"))

    return Trajectory(id=data.id, content=content)


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        standardized_data = process_data(raw_data)
        print(standardized_data.model_dump_json())
