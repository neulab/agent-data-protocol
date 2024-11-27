import sys
import json

from schema.action.api import ApiAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory
from schema_raw import SchemaRaw


def process_data(data):
    content = []
    for item in data["trajectory"]:
        if not item["source"]:
            item["source"] = "user" if item["action"] == "message" else "environment"
        if item["source"] == "agent" and item["message"].startswith("Running command: "):
            content.append(
                ApiAction(
                    function=item["tool_call_metadata"]["function_name"],
                    kwargs={
                        "command": item["message"].replace("Running command: ", "")
                    },
                )
            )
        else:
            content.append(
                TextObservation(
                    source=item["source"],
                    content=item["content"] if item["content"] else item["message"],
                )
            )

    return Trajectory(
        id=data["feedback_id"],
        content=content,
        details={
            "feedback": data["feedback"],
            "version": data["version"],
            "polarity": data["polarity"],
            "timestamp": data["timestamp"],
        },
    )


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data).model_dump()
        standardized_data = process_data(data)
        print(standardized_data.model_dump_json())