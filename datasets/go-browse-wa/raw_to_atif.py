# ruff: noqa: E402

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from schema.atif import Agent, ATIFObservation, ATIFTrajectory, ObservationResult, Step, ToolCall

GO_BROWSE_WA_VIEWPORT_SIZE = (1280, 1440)


def ast_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [ast_value(item) for item in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -ast_value(node.operand)
    raise ValueError(f"Unsupported action argument node: {type(node)}")


def parse_action(action_str: str) -> tuple[str, dict[str, Any]]:
    tree = ast.parse(action_str)
    if not isinstance(tree.body[0], ast.Expr) or not isinstance(tree.body[0].value, ast.Call):
        raise ValueError(f"Invalid action string: {action_str}")
    call = tree.body[0].value
    if not isinstance(call.func, ast.Name):
        raise ValueError(f"Invalid action function: {action_str}")
    function_name = call.func.id
    kwargs: dict[str, Any] = {}
    positional_names = {
        "click": ["bid"],
        "fill": ["bid", "value"],
        "select_option": ["bid", "options"],
        "scroll": ["delta_x", "delta_y"],
        "send_msg_to_user": ["text"],
        "noop": ["wait_ms"],
    }.get(function_name, [])
    for index, arg in enumerate(call.args):
        key = positional_names[index] if index < len(positional_names) else f"arg{index}"
        kwargs[key] = ast_value(arg)
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError(f"Unsupported **kwargs in action: {action_str}")
        kwargs[keyword.arg] = ast_value(keyword.value)
    return function_name, kwargs


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
    function_name, kwargs = parse_action(raw_step["step_data"]["parsed_action"])
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
    previous_action = ""
    for raw_step in raw_steps:
        function_name, _ = parse_action(raw_step["step_data"]["parsed_action"])
        if previous_action == "noop" and function_name == "noop":
            raise ValueError("consecutive noop")
        previous_action = function_name
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
            emit_trajectory(current_traj_id, current_goal, current_steps)
            current_steps = []
        current_traj_id = traj_id
        current_goal = traj_data["goal"]
        if traj_data.get("reward", 0) >= 1:
            current_steps.append(raw_step)
    if current_traj_id is not None and current_steps:
        emit_trajectory(current_traj_id, current_goal, current_steps)


if __name__ == "__main__":
    main()
