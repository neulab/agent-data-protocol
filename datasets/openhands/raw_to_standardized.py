import ast
import inspect
import json
import sys
import time

import api
import markdown
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str
from schema_raw import Args, SchemaRaw

from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.image import ImageObservation
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory

ACTION_MAP = {
    "fill": "type",
    "select_option": "select",
}

KWARGS_MAP = {
    "bid": "element_id",
    "value": "text",
    "delta_x": "dx",
    "delta_y": "dy",
    "options": "value",
}


# click('48', 'example with "quotes" and, a comma', 10, button='middle', modifiers=['Shift', 'Alt'])
def parse_browser_action(action_str):
    try:
        action_ast = ast.parse(action_str, mode="eval")
    except Exception:
        print(f"Invalid action string: {action_str}", file=sys.stderr)
        return None, [], {}
    if not isinstance(action_ast, ast.Expression) or not isinstance(action_ast.body, ast.Call):
        print(f"Invalid action string: {action_str}", file=sys.stderr)
        return None, [], {}
    call_node = action_ast.body
    function_name = call_node.func.id
    args = [ast.literal_eval(arg) for arg in call_node.args]
    kwargs = {
        KWARGS_MAP.get(kw.arg, kw.arg): ast.literal_eval(kw.value) for kw in call_node.keywords
    }
    return function_name, args, kwargs


def process_data(data, keep_all=False):
    content = []
    for item in data.trajectory:
        # In the huggingface dataset, args have been moved to extras
        if item.extras:
            item.args = Args(
                **{
                    **(item.args.dict() if item.args else {}),
                    **(item.extras.dict() if item.extras else {}),
                }
            )
        if not keep_all and item.source == "environment":
            continue
        if not item.action and (
            item.observation
            or item.log
            or item.message
            or item.content
            or item.error
            or item.error_code
            or item.status
        ):
            if item.observation == "browse":
                _html = (
                    flatten_dom_to_str(item.extras.dom_object) if item.extras.dom_object else None
                )
                if not _html and item.content.strip():
                    _html = markdown.markdown(item.content)
                content.append(
                    WebObservation(
                        source=item.source,
                        url=item.extras.url,
                        html=_html,
                        axtree=None
                        if not item.extras.axtree_object
                        else flatten_axtree_to_str(item.extras.axtree_object),
                        image_observation=None
                        if not item.extras.screenshot
                        else ImageObservation(
                            source=item.source,
                            content=item.extras.screenshot,  # Base64-encoded image data, not a path
                        ),
                        viewport_size=None,
                    )
                )
            elif item.observation == "run":
                obs = [item.message]
                obs += [f"Output:\n{item.content}\n"]
                content.append(
                    TextObservation(
                        source=item.source,
                        content="\n".join(obs),
                    )
                )
            elif item.observation == "run_ipython":
                obs = [item.message]
                obs += [f"Code:\n{item.extras.code}\n"]
                obs += [f"Output:\n{item.content}\n"]
                content.append(
                    TextObservation(
                        source=item.source,
                        content="\n".join(obs),
                    )
                )
            elif item.observation == "agent_state_changed":
                if not keep_all:
                    continue
                content.append(
                    TextObservation(
                        source=item.source,
                        content=f"Agent state changed to {item.extras.agent_state}",
                    )
                )
            elif item.observation == "delegate":
                content.append(
                    TextObservation(
                        source=item.source,
                        content=item.content if item.content else item.extras.outputs.content,
                    )
                )
            elif item.observation in [
                "edit",
                "write",
                "read",
                "rag_search",
                "crawl",
                "task_plan",
                "error",
                "user_rejected",
            ]:
                # avoid outputting consecutive TextObservations
                if (
                    not keep_all
                    and item.observation == "error"
                    and content
                    and isinstance(content[-1], Observation)
                ):
                    continue
                content.append(
                    TextObservation(
                        source=item.source,
                        content=f"{item.message}\n{item.content}"
                        if item.message != item.content
                        else item.message,
                    )
                )
            else:
                if not keep_all and (item.message == "No observation" or item.log is not None):
                    continue
                # just print all non-empty fields
                keys = ["observation", "message", "content", "log", "status", "error", "error_code"]
                obs = [
                    f"{k}: {getattr(item, k)}"
                    for k in keys
                    if getattr(item, k, None)
                    and not (isinstance(getattr(item, k), str) and not getattr(item, k).strip())
                ]
                print(f'Unknown observation: {"\n".join(obs)}', file=sys.stderr)
                content.append(
                    TextObservation(
                        source=item.source,
                        content="\n".join(obs),
                    )
                )
        elif item.action == "message":
            if not item.args.content:
                print("Empty message content, skipping!", file=sys.stderr)
                continue
            if item.source == "user":
                content.append(
                    TextObservation(
                        source=item.source,
                        content=item.args.content,
                    )
                )
            else:
                if item.args.content.startswith("USER (assistant): "):
                    item.args.content = item.args.content[len("USER (assistant): ") :]
                content.append(
                    MessageAction(
                        content=item.args.content,
                    )
                )
        elif item.action == "initialize":
            env_vars = {k: v for k, v in item.args.dict().items() if v is not None}
            content.append(
                ApiAction(
                    function=item.action,
                    kwargs={
                        "env_vars": env_vars,
                    },
                )
            )
        elif item.action == "run":
            content.append(
                ApiAction(
                    function=item.action,
                    description=item.args.thought,
                    kwargs={
                        "command": item.args.command,
                    },
                )
            )
        elif item.action == "run_ipython":
            content.append(
                ApiAction(
                    function=item.action,
                    description=item.args.thought,
                    kwargs={
                        "code": item.args.code,
                        "kernel_init_code": item.args.kernel_init_code,
                    },
                )
            )
        elif item.action == "browse_interactive":
            # fill('a12', 'example with "quotes" and, a comma')\nclick('a51')\nclick('48', button='middle', modifiers=['Shift', 'Alt'])
            action = item.args.browser_actions.strip()
            if not action:
                continue
            actions = [a.strip() for a in action.split("\n") if a.strip()]
            for action in actions:
                function_name, args, kwargs = parse_browser_action(action)
                if not function_name:
                    print(f"Invalid browser action: {action}", file=sys.stderr)
                    continue
                function_name = ACTION_MAP.get(function_name, function_name)
                api_args = list(inspect.signature(getattr(api, function_name)).parameters.keys())
                kwargs = {k: v for k, v in kwargs.items() if k in api_args}
                for arg in zip(args, api_args):
                    kwargs[arg[1]] = arg[0]
                content.append(
                    ApiAction(
                        function=function_name,
                        description=item.args.thought,
                        kwargs=kwargs,
                    )
                )
        elif item.action == "finish":
            if not keep_all:
                continue
            content.append(
                ApiAction(
                    function=item.action,
                    description=item.args.thought,
                    kwargs={
                        "output": item.args.outputs.content,
                    },
                )
            )
        elif item.action == "delegate":
            if not keep_all:
                continue
            if item.args.agent == "RagAgent":
                content.append(
                    ApiAction(
                        function="delegate_to_RagAgent",
                        description=item.args.thought,
                        kwargs={
                            "task": item.args.inputs.task,
                            "query": item.args.inputs.query,
                        },
                    )
                )
            elif item.args.agent == "CrawlAgent":
                content.append(
                    ApiAction(
                        function="delegate_to_CrawlAgent",
                        description=item.args.thought,
                        kwargs={
                            "task": item.args.inputs.task,
                            "link": item.args.inputs.link,
                        },
                    )
                )
            else:
                content.append(
                    ApiAction(
                        function="delegate_to_agent",
                        description=item.args.thought,
                        kwargs={
                            "agent": item.args.agent,
                            "task": item.args.inputs.task,
                        },
                    )
                )
        elif item.action == "add_task":
            content.append(
                ApiAction(
                    function=item.action,
                    description=item.args.thought,
                    kwargs={
                        "goal": item.args.goal,
                    },
                )
            )
        elif item.action == "modify_task":
            content.append(
                ApiAction(
                    function=item.action,
                    description=item.args.thought,
                    kwargs={
                        "task_id": item.args.task_id,
                        "state": item.args.state,
                    },
                )
            )
        elif item.action == "save_plan":
            content.append(
                ApiAction(
                    function=item.action,
                    description=item.args.thought,
                    kwargs={
                        "plan": item.args.plan,
                    },
                )
            )
        elif item.action == "task_plan":
            plan = item.args.plan
            if isinstance(plan, dict):
                plan = "\n".join(
                    [
                        f"Subtask {n}:\nDescription: {s["description"]}\nTool: {s["tool"]}"
                        for n, s in enumerate(plan["subtasks"], 1)
                    ]
                )
            content.append(
                ApiAction(
                    function=item.action,
                    description=item.args.thought,
                    kwargs={
                        "task": item.args.task,
                        "plan": plan,
                    },
                )
            )
        elif item.action == "read":
            content.append(
                ApiAction(
                    function=item.action,
                    description=item.args.thought,
                    kwargs={
                        "path": item.args.path,
                        "start": item.args.start,
                        "end": item.args.end,
                    },
                )
            )
        elif item.action == "edit" or item.action == "write":
            content.append(
                ApiAction(
                    function="edit",
                    description=item.args.thought,
                    kwargs={
                        "path": item.args.path,
                        "content": item.args.content,
                        "start": item.args.start,
                        "end": item.args.end,
                    },
                )
            )
        elif item.action == "crawl":
            content.append(
                ApiAction(
                    function=item.action,
                    description=item.args.thought,
                    kwargs={
                        "link": item.args.link,
                    },
                )
            )
        elif item.action == "rag_search":
            content.append(
                ApiAction(
                    function=item.action,
                    description=item.args.thought,
                    kwargs={
                        "query": item.args.query,
                    },
                )
            )
        elif item.action == "change_agent_state":
            if not keep_all:
                continue
            content.append(
                ApiAction(
                    function=item.action,
                    kwargs={
                        "agent_state": item.args.agent_state,
                    },
                )
            )
        else:
            print(f"Unknown action: {item.action}", file=sys.stderr)

    return Trajectory(
        id=str(time.time()),
        content=content,
        details={
            "feedback": data.feedback or "",
            "version": data.version or "",
            "polarity": data.polarity or "",
            "timestamp": data.timestamp or "",
        },
    )


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keep_all", action="store_true", help="Do not filter out any trajectory items"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data, keep_all=args.keep_all)
        print(standardized_data.model_dump_json())
