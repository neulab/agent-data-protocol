import sys
import json

from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory
from schema_raw import SchemaRaw


def process_data(data):
    content = []
    for msg in data.messages:
        if msg.role in ["system", "user", "tool"]:
            _msg = f"[{msg.name}] {msg.content}" if msg.role == "tool" else msg.content
            _msg = '\n'.join(_msg.split('] OBSERVATION:\n')[1:])
            content.append(
                TextObservation(
                    content=_msg,
                    source="user" if msg.role == "tool" else msg.role,
                )
            )
        elif msg.role == "assistant":
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call.type != "function":
                        print(f"Unknown tool call type: {tool_call.type}", file=sys.stderr)
                        continue
                    content.append(
                        ApiAction(
                            function=tool_call.function.name,
                            kwargs=json.loads(tool_call.function.arguments),
                        )
                    )
            else:
                content.append(MessageAction(content=msg.content))

    return Trajectory(
        id=data.instance_id,
        content=content,
        details={
            "run_id": data.run_id,
            "resolved": str(data.resolved),
            "tools": json.dumps(data.tools),
            "test_result": json.dumps(data.test_result),
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
