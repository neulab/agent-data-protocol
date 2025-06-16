import json
import random
import sys
import os

from schema_raw import SchemaRaw

from generate_thought import generate_thought

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

GENERATED_THOUGHTS_FILE = os.path.join(os.path.dirname(__file__), 'generated_thoughts.json')
if os.path.exists(GENERATED_THOUGHTS_FILE):
    with open(GENERATED_THOUGHTS_FILE) as f: 
        GENERATED_THOUGHTS = json.load(f)    
else:
    GENERATED_THOUGHTS = {}
def process_data(data):
    id = data.instance_id
    if id not in GENERATED_THOUGHTS:
        GENERATED_THOUGHTS[id] = {}
    print(f"{id}", file=sys.stderr)
    content = []
    parallel_tool_count = 0
    for idx, msg in enumerate(data.messages):
        idx = str(idx)
        if msg.role == "system":
            continue
        elif msg.role in ["user", "tool"]:
            _msg = f"{msg.content}" if msg.role == "tool" else msg.content
            if "OBSERVATION:\n" in _msg:
                _msg = "\n".join(_msg.split("OBSERVATION:\n")[1:])
            # Map the roles to the allowed source values in the schema
            source_map = {"user": "user", "tool": "environment"}
            _msg = TextObservation(
                content=_msg,
                source=source_map[msg.role],
            )
            if parallel_tool_count != 0:
                parallel_tool_count -= 1
            if parallel_tool_count == 0:
                content.append(_msg)
            else:
                # Handle parallel tool calls observations
                content = (
                    content[:(-parallel_tool_count)] + [_msg] + content[(-parallel_tool_count):]
                )
        elif msg.role == "assistant":
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call.type != "function":
                        print(f"Unknown tool call type: {tool_call.type}", file=sys.stderr)
                        continue
                    kwargs = json.loads(tool_call.function.arguments)
                    # Add required message parameter for finish function if not present
                    if tool_call.function.name == "finish":
                        if "message" not in kwargs:
                            kwargs["message"] = "Task completed."
                        content.append(
                            MessageAction(
                                content=f"<finish> {kwargs['message']} </finish>",
                                description=msg.content,
                            )
                        )
                    elif tool_call.function.name == "execute_bash":
                        parallel_tool_count += 1
                        thought = msg.content
                        if not thought:
                            if idx not in GENERATED_THOUGHTS[id]: 
                                context = []
                                for m in content:
                                    if isinstance(m, TextObservation):
                                        temp = TextObservation(
                                                content=m.content[:100] + ' ......[Truncated]',
                                                source=m.source
                                            )
                                        context.append(temp)
                                    else: context.append(m)
                                GENERATED_THOUGHTS[id][idx] = generate_thought(context, 'code_action', "bash", kwargs)
                            thought = GENERATED_THOUGHTS[id][idx]
                        content.append(
                            CodeAction(
                                language="bash",
                                content=kwargs["command"],
                                description=thought,
                            )
                        )
                    else:
                        parallel_tool_count += 1
                        thought = msg.content
                        if not thought:
                            context = []
                            if idx not in GENERATED_THOUGHTS[id]: 
                                for m in content:
                                    if isinstance(m, TextObservation):
                                        temp = TextObservation(
                                                content=m.content[:100] + ' ......[Truncated]',
                                                source=m.source
                                            )
                                        context.append(temp)
                                    else: context.append(m)
                                GENERATED_THOUGHTS[id][idx] = generate_thought(context, 'api_action', tool_call.function.name, kwargs)
                            thought = GENERATED_THOUGHTS[id][idx]
                            
                        content.append(
                            ApiAction(
                                description=thought,
                                function=tool_call.function.name,
                                kwargs=kwargs,
                            )
                        )
            else:
                content.append(MessageAction(content=msg.content))
        else:
            assert False
    if not isinstance(content[-1], MessageAction) or "<finish>" not in content[-1].content:
        user_end_message = random.choice(
            [
                [
                    TextObservation(
                        content="Congratulations! You have successfully solved the task.",
                        source="user",
                    ),
                ],
                [
                    TextObservation(
                        content="Your solution has been verified as correct. ", source="user"
                    ),
                ],
                [
                    TextObservation(
                        content="Well done on successfully completing the task!", source="user"
                    ),
                ],
                [
                    TextObservation(
                        content="Your implementation satisfies the task requirements.",
                        source="user",
                    ),
                ],
                [
                    TextObservation(content="Task completed successfully.", source="user"),
                ],
            ]
        )
        content.extend(user_end_message)
        assistant_end_message = random.choice(
            [
                [
                    MessageAction(
                        content="<finish> I have successfully completed the task. </finish>",
                        description="",
                    ),
                ],
                [
                    MessageAction(
                        content="<finish> I did it! The task is now complete. </finish>",
                        description="",
                    ),
                ],
                [
                    MessageAction(
                        content="<finish> The objective has been achieved with no outstanding issues. </finish>",
                        description="",
                    ),
                ],
                [
                    MessageAction(
                        content="<finish> I have fulfilled all the requirements of the task. </finish>",
                        description="",
                    ),
                ],
                [
                    MessageAction(
                        content="<finish> I've wrapped up the task successfully. </finish>",
                        description="",
                    ),
                ],
            ]
        )
        content.extend(assistant_end_message)
    with open(GENERATED_THOUGHTS_FILE, 'w') as f: 
        json.dump(GENERATED_THOUGHTS, f, indent=2, ensure_ascii=False)
    return Trajectory(
        id=id,
        content=content,
        details={
            "run_id": data.run_id,
            "resolved": str(data.resolved),
            "tools": json.dumps(data.tools, indent=2),
            "test_result": json.dumps(data.test_result, indent=2),
        },
    )


from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
    
if __name__ == "__main__":
    with open('/project/flame/yueqis/agent-data-collection/datasets/SWE-Gym_OpenHands-Sampled-Trajectories/full_raw.jsonl') as f:
        f = f.readlines()

    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        if not data.resolved:
            continue
        standardized_data = process_data(data)
        if standardized_data:
            print(standardized_data.model_dump_json())
