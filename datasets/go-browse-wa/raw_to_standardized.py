import ast
import json
import os
import sys
from typing import Any, Dict, Tuple

import api

from schema.action.api import ApiAction
from schema.observation.image import ImageObservation
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory

SCREENSHOTS_DIR = "datasets/go-browse-wa/screenshots"
GO_BROWSE_WA_VIEWPORT_SIZE = (1280, 1440)


def parse_action(action_str: str) -> Tuple[str, Dict[str, Any]]:
    """Parse an action string into function name and kwargs.

    Args:
        action_str (str): String representation of a function call (e.g., "click('231')", "noop(1000.5)")

    Returns:
        Tuple[str, Dict[str, Any]]: Function name and kwargs dictionary
    """
    # Parse the action string into an AST
    tree = ast.parse(action_str)
    if not isinstance(tree.body[0], ast.Expr) or not isinstance(tree.body[0].value, ast.Call):
        raise ValueError(f"Invalid action string: {action_str}")

    call = tree.body[0].value
    func_name = call.func.id

    func = getattr(api, func_name)
    param_names = func.__code__.co_varnames[: func.__code__.co_argcount]

    def eval_ast_node(node: ast.AST) -> Any:
        """Evaluate an AST node to its Python value."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.List):
            return [eval_ast_node(elt) for elt in node.elts]
        elif isinstance(node, ast.Name):
            if node.id == "True":
                return True
            elif node.id == "False":
                return False
            elif node.id == "None":
                return None
            else:
                raise ValueError(f"Unsupported name literal: {node.id}")
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -eval_ast_node(node.operand)
            else:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")

    # Convert positional args to kwargs
    kwargs = {}
    for i, arg in enumerate(call.args):
        if i < len(param_names):
            kwargs[param_names[i]] = eval_ast_node(arg)

    # Handle keyword arguments
    for kw in call.keywords:
        kwargs[kw.arg] = eval_ast_node(kw.value)

    return func_name, kwargs


def process_step(step):
    screenshot_message = ImageObservation(
        content=os.path.join(
            SCREENSHOTS_DIR,
            f"{step['traj_data']['traj_num']:05d}-{step['step_data']['step_number']:02d}.png",
        ),
        source="environment",
    )

    web_observation_message = WebObservation(
        axtree=step["step_data"]["obs"]["axtree_txt"],
        image_observation=screenshot_message,
        viewport_size=GO_BROWSE_WA_VIEWPORT_SIZE,
        url=None,
        html=None,
    )

    action, thought = step["step_data"]["parsed_action"], step["step_data"]["thought"]
    func_name, kwargs = parse_action(action)

    action_message = ApiAction(
        function=func_name,
        kwargs=kwargs,
        description=thought,
    )

    return [web_observation_message, action_message]


if __name__ == "__main__":
    traj_id = -1
    traj_content = []
    traj_goal = None

    for line in sys.stdin:
        step = json.loads(line)

        curr_traj_id = step["traj_data"]["traj_num"]

        if traj_id != -1 and traj_id != curr_traj_id and traj_content:
            goal_message = TextObservation(content=traj_goal, source="user")
            traj_content = [goal_message] + traj_content

            traj = Trajectory(
                id=str(traj_id),
                content=traj_content,
                details={"source": "go-browse-wa"},
            )
            print(json.dumps(traj.model_dump()))
            traj_content = []

        if step["traj_data"]["reward"] < 1:
            traj_id = curr_traj_id
            continue

        traj_goal = step["traj_data"]["goal"]
        traj_content.extend(process_step(step))
        traj_id = curr_traj_id

    if traj_content:
        goal_message = TextObservation(content=traj_goal, source="user")
        traj_content = [goal_message] + traj_content

        traj = Trajectory(
            id=str(traj_id),
            content=traj_content,
            details={"source": "go-browse-wa"},
        )
        print(json.dumps(traj.model_dump()))
