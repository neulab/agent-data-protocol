# ruff: noqa: E402

from __future__ import annotations

import ast
import json
import re
import sys
from typing import Any

from schema.atif import Agent, ATIFObservation, ATIFTrajectory, ObservationResult, Step, ToolCall

GO_BROWSE_WA_VIEWPORT_SIZE = (1280, 1440)
POSITIONAL_ACTION_ARGS = {
    "click": ["bid"],
    "fill": ["bid", "value"],
    "select_option": ["bid", "options"],
    "scroll": ["delta_x", "delta_y"],
    "send_msg_to_user": ["text"],
    "noop": ["wait_ms"],
}


def ast_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [ast_value(item) for item in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -ast_value(node.operand)
    raise ValueError(f"Unsupported action argument node: {type(node)}")


def parse_action(action_str: str) -> tuple[str, dict[str, Any]]:
    try:
        tree = ast.parse(action_str)
    except SyntaxError:
        recovered = parse_malformed_action(action_str)
        if recovered is not None:
            return recovered
        raise
    if not isinstance(tree.body[0], ast.Expr) or not isinstance(tree.body[0].value, ast.Call):
        raise ValueError(f"Invalid action string: {action_str}")
    call = tree.body[0].value
    if not isinstance(call.func, ast.Name):
        raise ValueError(f"Invalid action function: {action_str}")
    function_name = call.func.id
    kwargs: dict[str, Any] = {}
    positional_names = POSITIONAL_ACTION_ARGS.get(function_name, [])
    for index, arg in enumerate(call.args):
        key = positional_names[index] if index < len(positional_names) else f"arg{index}"
        kwargs[key] = ast_value(arg)
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError(f"Unsupported **kwargs in action: {action_str}")
        kwargs[keyword.arg] = ast_value(keyword.value)
    return function_name, kwargs


def parse_malformed_action(action_str: str) -> tuple[str, dict[str, Any]] | None:
    match = re.match(r"^([A-Za-z_]\w*)\((.*)\)\}?$", action_str.strip(), flags=re.DOTALL)
    if not match:
        return None
    function_name = match.group(1)
    positional_names = POSITIONAL_ACTION_ARGS.get(function_name)
    if not positional_names:
        return None
    args_text = match.group(2)
    if function_name == "send_msg_to_user":
        raw_args = [args_text.strip()] if args_text else []
    elif function_name in {"click", "noop"}:
        raw_args = [args_text.split(",", 1)[0].strip()] if args_text else []
    else:
        max_splits = max(len(positional_names) - 1, 0)
        raw_args = [part.strip() for part in args_text.split(",", max_splits)] if args_text else []
    kwargs = {}
    for index, raw_arg in enumerate(raw_args[: len(positional_names)]):
        kwargs[positional_names[index]] = parse_malformed_arg(raw_arg)
    return function_name, kwargs


def parse_malformed_arg(raw_arg: str) -> Any:
    try:
        return ast.literal_eval(raw_arg)
    except (SyntaxError, ValueError):
        value = raw_arg.strip()
        if value[:1] in {"'", '"'}:
            value = value[1:]
        if value[-1:] in {"'", '"'}:
            value = value[:-1]
        return value


def get_action_string(raw_step: dict[str, Any]) -> str:
    step_data = raw_step["step_data"]
    parsed_action = step_data.get("parsed_action")
    if isinstance(parsed_action, str) and parsed_action.strip():
        try:
            parse_action(parsed_action)
            return parsed_action
        except (SyntaxError, ValueError):
            pass

    raw_action = step_data.get("action")
    if isinstance(raw_action, str):
        try:
            action_data = json.loads(raw_action)
        except json.JSONDecodeError:
            action_data = None
        if isinstance(action_data, dict):
            action_value = action_data.get("action")
            if isinstance(action_value, str):
                return action_value
            if isinstance(action_value, dict) and len(action_value) == 1:
                function_name, arguments = next(iter(action_value.items()))
                if isinstance(arguments, dict):
                    kwargs = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
                    return f"{function_name}({kwargs})"

        match = re.search(r'"action"\s*:\s*("(?:\\.|[^"\\])*")', raw_action)
        if match:
            action_value = json.loads(match.group(1))
            try:
                parse_action(action_value)
                return action_value
            except (SyntaxError, ValueError):
                pass
        marker = '"action": "'
        marker_index = raw_action.rfind(marker)
        if marker_index >= 0:
            action_value = raw_action[marker_index + len(marker) :].strip()
            if action_value.endswith("}"):
                action_value = action_value[:-1].strip()
            if action_value.endswith('"'):
                action_value = action_value[:-1].strip()
            try:
                parse_action(action_value)
                return action_value
            except (SyntaxError, ValueError):
                pass

    raise ValueError(
        "missing parsed action for "
        f"trajectory {raw_step.get('traj_data', {}).get('traj_num')} "
        f"step {step_data.get('step_number')}"
    )


def make_observation_step(step_id: int, raw_step: dict[str, Any]) -> Step:
    step_data = raw_step["step_data"]
    return Step(
        step_id=step_id,
        source="agent",
        message="",
        observation=ATIFObservation(
            results=[
                ObservationResult(
                    content="",
                    extra={
                        "web": {
                            "axtree": step_data["obs"].get("axtree_txt"),
                            "html": None,
                            "url": raw_step.get("node_data", {}).get("node_url"),
                            "viewport_size": list(GO_BROWSE_WA_VIEWPORT_SIZE),
                            "image_observation": {
                                "content": (
                                    "datasets/go-browse-wa/screenshots/"
                                    f"{raw_step['traj_data']['traj_num']:05d}-"
                                    f"{step_data['step_number']:02d}.png"
                                ),
                                "source": "environment",
                            },
                        }
                    },
                )
            ]
        ),
    )


def make_action_step(step_id: int, raw_step: dict[str, Any], call_id: str) -> Step:
    function_name, kwargs = parse_action(get_action_string(raw_step))
    thought = raw_step["step_data"].get("thought")
    if function_name == "send_msg_to_user":
        function_name = "finish"
        kwargs = {"message": kwargs.get("text", ""), "task_completed": "true"}
    return Step(
        step_id=step_id,
        source="agent",
        message=thought or "",
        tool_calls=[
            ToolCall(
                tool_call_id=call_id,
                function_name=function_name,
                arguments=kwargs,
            )
        ],
    )


def emit_trajectory(traj_id: int, goal: str, raw_steps: list[dict[str, Any]]) -> None:
    steps = [Step(step_id=1, source="user", message=goal)]
    next_step_id = 2
    next_call_id = 1
    for raw_step in raw_steps:
        steps.append(make_observation_step(next_step_id, raw_step))
        next_step_id += 1
        steps.append(make_action_step(next_step_id, raw_step, f"call_{next_call_id:06d}"))
        next_step_id += 1
        next_call_id += 1
    if not steps[-1].tool_calls or steps[-1].tool_calls[0].function_name != "finish":
        raise ValueError(f"trajectory {traj_id} did not complete")
    trajectory = ATIFTrajectory(
        trajectory_id=str(traj_id),
        agent=Agent(name="go-browse-wa", version="atif"),
        steps=steps,
        extra={"source_dataset": "go-browse-wa"},
    )
    print(trajectory.model_dump_json(exclude_none=True))


def safe_emit_trajectory(traj_id: int, goal: str, raw_steps: list[dict[str, Any]]) -> None:
    # Consecutive noop actions are valid wait events in this source data; only
    # skip trajectories that are malformed or incomplete.
    try:
        emit_trajectory(traj_id, goal, raw_steps)
    except Exception as exc:
        print(f"Skipping trajectory {traj_id}: {exc}", file=sys.stderr)


def main() -> None:
    current_traj_id: int | None = None
    current_goal = ""
    current_steps: list[dict[str, Any]] = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        raw_step = json.loads(line)
        traj_data = raw_step["traj_data"]
        traj_id = int(traj_data["traj_num"])
        if current_traj_id is not None and traj_id != current_traj_id and current_steps:
            safe_emit_trajectory(current_traj_id, current_goal, current_steps)
            current_steps = []
        current_traj_id = traj_id
        current_goal = traj_data["goal"]
        if traj_data.get("reward", 0) >= 1:
            current_steps.append(raw_step)
    if current_traj_id is not None and current_steps:
        safe_emit_trajectory(current_traj_id, current_goal, current_steps)


if __name__ == "__main__":
    main()
