import json
import re
import sys

from schema_raw import SchemaRaw

from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

COMMAND_BLOCK_RE = re.compile(r"```(?:bash|sh)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
THOUGHT_PREFIX_RE = re.compile(r"^\s*THOUGHT:\s*", re.IGNORECASE)


def parse_assistant_message(content: str) -> CodeAction | MessageAction:
    match = COMMAND_BLOCK_RE.search(content)
    if not match:
        return MessageAction(content=content)

    command = match.group(1).strip()
    description_parts = [content[: match.start()].strip(), content[match.end() :].strip()]
    description = "\n\n".join(part for part in description_parts if part)
    description = THOUGHT_PREFIX_RE.sub("", description).strip()

    return CodeAction(language="bash", content=command, description=description or None)


def process_data(data: SchemaRaw) -> Trajectory:
    content = []
    seen_initial_user = False

    for message in data.messages:
        if message.role == "system":
            continue
        if message.role == "user":
            source = "environment" if seen_initial_user else "user"
            content.append(TextObservation(content=message.content, source=source))
            seen_initial_user = True
        elif message.role == "assistant":
            content.append(parse_assistant_message(message.content))
        else:
            raise ValueError(f"Invalid role: {message.role}")

    has_submission = any(
        isinstance(item, CodeAction) and "MINI_SWE_AGENT_FINAL_OUTPUT" in item.content
        for item in content
    )
    if has_submission:
        content.append(TextObservation(content="Task completed successfully.", source="user"))
        content.append(
            MessageAction(
                content="<finish> I have completed the task successfully. </finish>",
                description="",
            )
        )

    return Trajectory(
        id=data.id,
        content=content,
        details={"instance_id": data.instance_id},
    )


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        print(standardized_data.model_dump_json())
