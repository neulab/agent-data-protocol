import sys
import json

from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory
from schema_raw import SchemaRaw



if __name__ == "__main__":

    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)

        if data.fail:
            continue

        content = []

        system_msg = None
        for message in data.messages:
            if message.role == "system":
                system_msg = message.content
            elif message.role == "user":
                if system_msg:
                    # prepend system message to the first user message
                    text = system_msg + " " + message.content
                    system_msg = None
                else:
                    text = message.content
                content.append(TextObservation(content=text, source=message.role))
            elif message.role == "assistant":
                content.append(MessageAction(content=message.content))

        standardized_data = Trajectory(
            id=data.instance_id,
            content=content,
            details={
                "exp_name": data.exp_name,
                "fail": str(data.fail),
            },
        )
        print(standardized_data.model_dump_json())