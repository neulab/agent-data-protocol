import ast
import inspect
import json
import random
import sys

import api
import markdown
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str
from schema_raw import Args, SchemaRaw

from schema.action.action import Action
from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.image import ImageObservation
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory


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
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call_node.keywords}
    return function_name, args, kwargs


def map_source(source):
    """Map source to one of the allowed values: 'user', 'agent', or 'environment'"""
    if source == "user":
        return "user"
    elif source == "assistant":
        return "agent"
    elif source == "system":
        return "environment"
    else:
        return "environment"


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
                        source=map_source(item.source),
                        url=item.extras.url,
                        html=_html,
                        axtree=None
                        if not item.extras.axtree_object
                        else flatten_axtree_to_str(item.extras.axtree_object),
                        image_observation=None
                        if not item.extras.screenshot
                        else ImageObservation(
                            source=map_source(item.source),
                            content=item.extras.screenshot,  # Base64-encoded image data, not a path
                        ),
                        viewport_size=None,
                    )
                )
            elif item.observation == "run":
                obs = f"{item.content}"
                if "\r\n" in obs:
                    obs = "\r\n".join(obs.split("\r\n")[1:])
                content.append(
                    TextObservation(
                        source=map_source(item.source),
                        content=obs,
                    )
                )
            elif item.observation == "run_ipython":
                obs = [f"{item.content}"]
                content.append(
                    TextObservation(
                        source=map_source(item.source),
                        content="\n".join(obs),
                    )
                )
            elif item.observation == "agent_state_changed":
                if not keep_all:
                    continue
                content.append(
                    TextObservation(
                        source=map_source(item.source),
                        content=f"Agent state changed to {item.extras.agent_state}",
                    )
                )
            elif item.observation == "delegate":
                content.append(
                    TextObservation(
                        source=map_source(item.source),
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
                        source=map_source(item.source),
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
                print(f"Unknown observation: {'\n'.join(obs)}", file=sys.stderr)
                content.append(
                    TextObservation(
                        source=map_source(item.source),
                        content="\n".join(obs),
                    )
                )
        if item.action == "message":
            if not item.args.content:
                print("Empty message content, skipping!", file=sys.stderr)
                continue
            if item.source == "user":
                # if item.args.content.lower() == "continue":
                #     continue
                content.append(
                    TextObservation(
                        source=map_source(item.source),
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
                CodeAction(
                    language="bash",
                    content=item.args.command,
                    description=item.args.thought,
                )
            )
        elif item.action == "run_ipython":
            content.append(
                CodeAction(
                    language="python",
                    content=item.args.code,
                    description=item.args.thought,
                )
            )
        elif item.action == "browse_interactive":
            # fill('a12', 'example with "quotes" and, a comma')\nclick('a51')\nclick('48', button='middle', modifiers=['Shift', 'Alt'])
            action = item.args.browser_actions.strip()
            if not action:
                return None
            actions = [a.strip() for a in action.split("\n") if a.strip()]
            for action in actions:
                function_name, args, kwargs = parse_browser_action(action)
                if not function_name:
                    print(f"Invalid browser action: {action}", file=sys.stderr)
                    continue
                try:
                    api_args = inspect.signature(getattr(api, function_name)).parameters
                except:
                    continue
                action_kwargs = {}
                for param, arg in zip(api_args.items(), args):
                    param_name, param_obj = param
                    if param_obj.annotation is str:
                        arg = f'"{arg}"'
                    action_kwargs[param_name] = arg
                content.append(
                    ApiAction(
                        function=function_name,
                        description=item.args.thought,
                        kwargs=action_kwargs,
                    )
                )
        elif item.action == "finish":
            if item.args.thought:
                thought = item.args.thought
            elif item.message:
                thought = item.message
            if item.args.outputs.content:
                output = item.args.outputs.content
            elif item.message:
                output = item.message
            else:
                output = None
            content.append(
                ApiAction(
                    function=item.action,
                    description=thought,
                    kwargs={
                        "output": f'"{output}"',
                    },
                )
            )
        elif item.action == "delegate":
            if item.args.agent == "RagAgent":
                content.append(
                    ApiAction(
                        function="delegate_to_RagAgent",
                        description=item.args.thought,
                        kwargs={
                            "task": f'"{item.args.inputs.task}"',
                            "query": f'"{item.args.inputs.query}"',
                        },
                    )
                )
            elif item.args.agent == "CrawlAgent":
                content.append(
                    ApiAction(
                        function="delegate_to_CrawlAgent",
                        description=item.args.thought,
                        kwargs={
                            "task": f'"{item.args.inputs.task}"',
                            "link": f'"{item.args.inputs.link}"',
                        },
                    )
                )
            else:
                content.append(
                    ApiAction(
                        function="delegate_to_agent",
                        description=item.args.thought,
                        kwargs={
                            "agent": f'"{item.args.agent}"',
                            "task": f'"{item.args.inputs.task}"',
                        },
                    )
                )
            content.append(TextObservation(source="user", content="Continue"))
        elif item.action == "add_task":
            content.append(
                ApiAction(
                    function=item.action,
                    description=item.args.thought,
                    kwargs={
                        "goal": f'"{item.args.goal}"',
                    },
                )
            )
        elif item.action == "modify_task":
            content.append(
                ApiAction(
                    function=item.action,
                    description=item.args.thought,
                    kwargs={
                        "task_id": f'"{item.args.task_id}"',
                        "state": f'"{item.args.state}"',
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
                        f"Subtask {n}:\nDescription: {s['description']}\nTool: {s['tool']}"
                        for n, s in enumerate(plan["subtasks"], 1)
                    ]
                )
            content.append(
                ApiAction(
                    function=item.action,
                    description=item.args.thought,
                    kwargs={
                        "task": f'"{item.args.task}"',
                        "plan": f'"{plan}"',
                    },
                )
            )
        elif item.action == "read":
            content.append(
                ApiAction(
                    function=item.action,
                    description=item.args.thought,
                    kwargs={
                        "path": f'"{item.args.path}"',
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
                        "path": f'"{item.args.path}"',
                        "content": f'"{item.args.content}"',
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
                        "link": f'"{item.args.link}"',
                    },
                )
            )
        elif item.action == "rag_search":
            content.append(
                ApiAction(
                    function=item.action,
                    description=item.args.thought,
                    kwargs={
                        "query": f'"{item.args.query}"',
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
                        "agent_state": f'"{item.args.agent_state}"',
                    },
                )
            )
        else:
            print(f"Unknown action: {item.action}", file=sys.stderr)

        # Combine consecutive agent message + action
        if (
            len(content) >= 2
            and isinstance(content[-1], Action)
            and isinstance(content[-2], MessageAction)
        ):
            pre_message = content.pop(-2).content
            if pre_message:
                if content[-1].description:
                    content[-1].description = pre_message + "\n\n" + content[-1].description
                else:
                    content[-1].description = pre_message
                content[-1].description = content[-1].description.strip()

        # Combine consecutive user message
        if (
            len(content) >= 2
            and isinstance(content[-1], TextObservation)
            and isinstance(content[-2], TextObservation)
        ):
            if content[-1].source != content[-2].source:
                return None
            pre_message = content.pop(-2).content
            if pre_message:
                if content[-1].content:
                    content[-1].content = pre_message + "\n\n" + content[-1].content
                else:
                    content[-1].content = pre_message
                content[-1].content = content[-1].content.strip()

        if (
            len(content) >= 2
            and isinstance(content[-1], Action)
            and isinstance(content[-2], Action)
        ):
            if isinstance(content[-1], MessageAction):
                continue
            return None

    #
    if not isinstance(content[0], Observation):
        return None
    # Handle Finish Message
    user_end_message = random.choice(
        [
            "Congratulations! You have successfully solved the task.",
            "Your solution has been verified as correct. ",
            "Well done on successfully completing the task!",
            "Your implementation satisfies the task requirements.",
            "Task completed successfully.",
        ]
    )
    assistant_end_message = random.choice(
        [
            "<finish> I have successfully completed the task. </finish>",
            "<finish> I did it! The task is now complete. </finish>",
            "<finish> The objective has been achieved with no outstanding issues. </finish>",
            "<finish> I have fulfilled all the requirements of the task. </finish>",
            "<finish> I've wrapped up the task successfully. </finish>",
        ]
    )
    if isinstance(content[-1], MessageAction) and not isinstance(content[-2], Observation):
        user_end_obs = TextObservation(
            content=user_end_message,
            source="user",
        )
        content = content[:-1] + [user_end_obs] + [content[-1]]
        content[-1].description = content[-1].content
        content[-1].content = assistant_end_message
    elif isinstance(content[-1], Observation):
        content.append(
            MessageAction(
                content=assistant_end_message,
                description="",
            ),
        )
    else:
        content.append(
            TextObservation(
                content=user_end_message,
                source="user",
            )
        )
        content.append(
            MessageAction(
                content=assistant_end_message,
                description="",
            ),
        )

    return Trajectory(
        id=data.id,
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

    # Process each line of input individually
    for line in sys.stdin:
        raw_data = json.loads(line)
        if raw_data["feedback"] != "positive":
            continue
        # if raw_data["id"] == "249": continue
        # if raw_data["id"] != "100": continue
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data, keep_all=args.keep_all)

        # Print the standardized data as JSON
        if standardized_data:
            print(json.dumps(standardized_data.model_dump()))
