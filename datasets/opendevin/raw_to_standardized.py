import sys
import json

import time
from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory
from schema_raw import SchemaRaw


def process_data(data):
    content = []
    for item in data.trajectory:
        if not item.action and (item.observation or item.log or item.message or item.content or item.error or item.error_code or item.status):
            obs = []
            keys = ["observation", "log", "message", "content", "error", "error_code", "status"]
            obs = [f"{k}: {getattr(item, k)}" for k in keys if getattr(item, k, None)]
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
            content.append(
                ApiAction(
                    function=item.action,
                    description=item.args.thought,
                    kwargs={
                        "browser_actions": item.args.browser_actions,
                    },
                )
            )
        elif item.action == "finish":
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
                plan = "\n".join([f"Subtask {n}:\nDescription: {s["description"]}\nTool: {s["tool"]}" for n, s in enumerate(plan["subtasks"], 1)])
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
            "feedback": data.feedback,
            "version": data.version,
            "polarity": data.polarity,
            "timestamp": data.timestamp,
        },
    )


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        print(standardized_data.model_dump_json())