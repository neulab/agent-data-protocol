import json
import sys

from schema_raw import SchemaRaw

from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


def process_data(data):
    content = []
    parallel_tool_count = 0
    for msg in data.messages:
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
                    else:
                        parallel_tool_count += 1
                        content.append(
                            ApiAction(
                                description=msg.content,
                                function=tool_call.function.name,
                                kwargs=kwargs,
                            )
                        )
            else:
                content.append(MessageAction(content=msg.content))
        else:
            assert False
    return Trajectory(
        id=data.instance_id,
        content=content,
        details={
            "run_id": data.run_id,
            "resolved": str(data.resolved),
            "tools": json.dumps(data.tools, indent=2),
            "test_result": json.dumps(data.test_result, indent=2),
        },
    )


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        if not data.resolved:
            continue
        standardized_data = process_data(data)
        print(standardized_data.model_dump_json())
