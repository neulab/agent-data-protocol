import ast
import inspect
import json
import os
import sys
from typing import Any, Dict, Literal, Tuple, Union, get_args, get_origin

import api

from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.image import ImageObservation
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory

SCREENSHOTS_DIR = "datasets/go-browse-wa/screenshots"
GO_BROWSE_WA_VIEWPORT_SIZE = (1280, 1440)


def send_msg_to_user(text: str) -> None:
    """Send a message to the user.

    Args:
    ----
        text (str): The message to send to the user.

    """
    pass


def parse_action(action_str: str) -> Tuple[str, Dict[str, Any]]:
    tree = ast.parse(action_str)
    if not isinstance(tree.body[0], ast.Expr) or not isinstance(tree.body[0].value, ast.Call):
        raise ValueError(f"Invalid action string: {action_str}")

    call = tree.body[0].value
    func_name = call.func.id

    if func_name == "send_msg_to_user":
        func = send_msg_to_user
    else:
        func = getattr(api, func_name)
    sig = inspect.signature(func)

    def eval_ast_node(node: ast.AST) -> Any:
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

    def should_quote(param_name: str) -> bool:
        param = sig.parameters.get(param_name)
        if param is None or param.annotation is inspect._empty:
            return False

        ann = param.annotation
        origin = get_origin(ann)
        args = get_args(ann)

        # Direct str
        if ann is str:
            return True

        # Literal[str, ...]
        if origin is Literal:
            return True

        # Union including str
        if origin is Union and any(a is str for a in args):
            return True

        # List[Literal[str, ...]]
        if origin is list and len(args) == 1:
            inner_origin = get_origin(args[0])
            inner_args = get_args(args[0])
            if inner_origin is Literal and all(isinstance(a, str) for a in inner_args):
                return True

        return False

    def quote(v):
        if isinstance(v, str):
            return f'"{v}"'
        if isinstance(v, list):
            return [quote(x) for x in v]
        return v

    kwargs = {}
    for i, arg in enumerate(call.args):
        if i < len(sig.parameters):
            param_name = list(sig.parameters.keys())[i]
            val = eval_ast_node(arg)
            if should_quote(param_name):
                val = quote(val)
            kwargs[param_name] = val

    for kw in call.keywords:
        val = eval_ast_node(kw.value)
        if should_quote(kw.arg):
            val = quote(val)
        kwargs[kw.arg] = val

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
    if func_name == "send_msg_to_user":
        action_message = MessageAction(
            content=kwargs["text"], description=thought.replace("send_msg_to_user", "finish")
        )
    else:
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
            try:
                goal_message = TextObservation(content=traj_goal, source="user")
                traj_content = [goal_message] + traj_content
                priot_action = ""
                for m in traj_content:
                    if isinstance(m, ApiAction):
                        if priot_action == "noop" and m.function == "noop":
                            raise ValueError("consecutive noop")
                        priot_action = m.function
                if not isinstance(traj_content[-1], MessageAction):
                    raise ValueError(f"trajectory did not complete: {traj_content[-1]}")
                traj_content[-1].content = f"<finish> {traj_content[-1].content} </finish>"
                traj = Trajectory(
                    id=str(traj_id),
                    content=traj_content,
                    details={"source": "go-browse-wa"},
                )
                print(json.dumps(traj.model_dump()))
                traj_content = []
            except Exception as e:
                print(f"An error occurred: {e}", file=sys.stderr)
                traj_id = -1
                traj_content = []
                traj_goal = None

        if step["traj_data"]["reward"] < 1:
            traj_id = curr_traj_id
            continue

        traj_goal = step["traj_data"]["goal"]
        try:
            traj_content.extend(process_step(step))
        except Exception as e:
            print(f"Failed to process step: {e}\n", file=sys.stderr)
            traj_id = -1
            traj_content = []
            traj_goal = None
            continue
        traj_id = curr_traj_id

    if traj_content:
        goal_message = TextObservation(content=traj_goal, source="user")
        traj_content = [goal_message] + traj_content
        priot_action = ""
        for m in traj_content:
            if isinstance(m, ApiAction):
                if priot_action == "noop" and m.function == "noop":
                    raise ValueError("consecutive noop")
                priot_action = m.function
        if not isinstance(traj_content[-1], MessageAction):
            raise ValueError(f"trajectory did not complete: {traj_content[-1]}")
        traj_content[-1].content = f"<finish> {traj_content[-1].content} </finish>"
        traj = Trajectory(
            id=str(traj_id),
            content=traj_content,
            details={"source": "go-browse-wa"},
        )
        print(json.dumps(traj.model_dump()))
