#!/usr/bin/env python3
"""Convert AgentNet raw data to ADP standardized format.

Reads raw AgentNet JSON from stdin (one trajectory per line), parses PyAutoGUI
code strings into structured ApiActions, and outputs Trajectory objects.

Usage:
    cat sample_raw.json | python scripts/json_to_jsonl.py | \
        python datasets/agentnet/raw_to_standardized.py | \
        python scripts/jsonl_to_json.py > sample_std.json
"""

import ast
import json
import sys
from typing import Any

sys.path.append(".")

from schema.action.api import ApiAction
from schema.observation.image import ImageObservation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

from schema_raw import SchemaRaw

# Maps PyAutoGUI function names to their positional parameter names.
# Used to convert positional args to named kwargs for the SFT converter.
# Names match PyAutoGUI's actual API (camelCase) and actual param names.
# None means variadic args, collected into a "keys" list.
POSITIONAL_PARAMS: dict[str, list[str] | None] = {
    "click": ["x", "y"],
    "doubleClick": ["x", "y"],
    "rightClick": ["x", "y"],
    "tripleClick": ["x", "y"],
    "middleClick": ["x", "y"],
    "write": ["message"],
    "press": ["key"],
    "hotkey": None,
    "scroll": ["clicks"],
    "hscroll": ["clicks"],
    "moveTo": ["x", "y"],
    "dragTo": ["x", "y"],
    "wait": [],
    "terminate": ["status"],
}


def _eval_ast_value(node: ast.expr) -> Any:
    """Extract a Python literal value from an AST node.

    Handles constants, negative numbers (UnaryOp), and lists.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _eval_ast_value(node.operand)
        return -operand
    if isinstance(node, ast.List):
        return [_eval_ast_value(elt) for elt in node.elts]
    raise ValueError(f"Unsupported AST node type: {type(node).__name__}")


def parse_pyautogui_call(code: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse a PyAutoGUI code string into (function_name, kwargs) tuples.

    Handles single-line calls like `pyautogui.click(x=0.16, y=0.27)` and
    multi-line code blocks with multiple statements.

    Args:
        code: Raw PyAutoGUI code string from AgentNet.

    Returns:
        List of (function_name, kwargs_dict) tuples.
    """
    code = code.strip()
    if not code:
        return []

    # Try single expression first, fall back to multi-statement
    try:
        tree = ast.parse(code, mode="eval")
        calls = [tree.body]
    except SyntaxError:
        tree = ast.parse(code, mode="exec")
        calls = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        ]

    results = []
    for call in calls:
        if not isinstance(call, ast.Call):
            continue

        # Extract function name (handles pyautogui.click and plain click)
        if isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        elif isinstance(call.func, ast.Name):
            func_name = call.func.id
        else:
            continue

        # Build kwargs from keyword arguments
        kwargs: dict[str, Any] = {}
        for kw in call.keywords:
            if kw.arg is not None:
                kwargs[kw.arg] = _eval_ast_value(kw.value)

        # Convert positional args to named kwargs
        param_names = POSITIONAL_PARAMS.get(func_name)
        if param_names is None and func_name in POSITIONAL_PARAMS:
            # Variadic function (hotkey): collect all positional args into "keys"
            kwargs["keys"] = [_eval_ast_value(arg) for arg in call.args]
        elif param_names is not None:
            for i, arg in enumerate(call.args):
                if i < len(param_names):
                    kwargs[param_names[i]] = _eval_ast_value(arg)

        results.append((func_name, kwargs))

    return results


def convert_to_trajectory(raw_data: dict[str, Any]) -> Trajectory:
    """Convert a raw AgentNet trajectory to ADP standardized format.

    Produces a sequence of:
      TextObservation (task instruction) →
      [ImageObservation → ApiAction] × N steps

    Args:
        raw_data: Raw trajectory dict from AgentNet JSONL.

    Returns:
        Standardized Trajectory object.
    """
    data = SchemaRaw(**raw_data)
    content = []

    # Task instruction as first user observation
    instruction = data.instruction or data.natural_language_task or ""
    content.append(TextObservation(content=instruction, source="user"))

    for step in data.traj:
        # Screenshot observation
        if step.image:
            content.append(
                ImageObservation(
                    content=f"datasets/agentnet/screenshots/{step.image}",
                    annotations=None,
                    source="user",
                )
            )

        # Parse PyAutoGUI code into structured action(s)
        code = step.value.code
        if code:
            thought = step.value.thought or ""
            try:
                parsed_actions = parse_pyautogui_call(code)
                for func_name, kwargs in parsed_actions:
                    content.append(
                        ApiAction(
                            function=func_name,
                            kwargs=kwargs,
                            description=thought,
                        )
                    )
                    # Only attach thought to the first action in multi-action blocks
                    thought = ""
            except (SyntaxError, ValueError) as e:
                print(
                    f"Warning: Could not parse code for {data.task_id} "
                    f"step {step.index}: {e}",
                    file=sys.stderr,
                )

    # Store metadata in details (Trajectory.details expects dict[str, str])
    details: dict[str, str] = {
        "dataset": "agentnet",
    }
    if data.task_completed is not None:
        details["task_completed"] = str(data.task_completed)
    if data.alignment_score is not None:
        details["alignment_score"] = str(data.alignment_score)
    if data.efficiency_score is not None:
        details["efficiency_score"] = str(data.efficiency_score)
    if data.task_difficulty is not None:
        details["task_difficulty"] = str(data.task_difficulty)

    return Trajectory(
        id=data.task_id,
        content=content,
        details=details,
    )


record_count = 0
error_count = 0
for line in sys.stdin:
    try:
        raw_data = json.loads(line)
        trajectory = convert_to_trajectory(raw_data)
        print(trajectory.model_dump_json())
        record_count += 1
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        error_count += 1
        print(f"Warning: Skipping record: {e}", file=sys.stderr)

print(f"Processed {record_count} trajectories ({error_count} errors)", file=sys.stderr)
