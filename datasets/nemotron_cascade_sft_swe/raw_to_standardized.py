import json
import re
import sys

from schema_raw import SchemaRaw

from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

THINK_BLOCK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)


def split_think_blocks(content: str) -> tuple[str, str | None]:
    think_blocks = [match.strip() for match in THINK_BLOCK_RE.findall(content)]
    visible_content = THINK_BLOCK_RE.sub("", content).strip()
    reasoning_content = "\n\n".join(block for block in think_blocks if block)
    return visible_content, reasoning_content or None


def convert_message(role: str, content: str):
    if role == "system":
        return None
    if role == "user":
        return TextObservation(content=content, source="user")
    if role == "assistant":
        visible_content, reasoning_content = split_think_blocks(content)
        return MessageAction(content=visible_content, reasoning_content=reasoning_content)
    raise ValueError(f"Unsupported message role: {role}")


def process_data(data: SchemaRaw) -> Trajectory:
    content = []
    for message in data.messages:
        event = convert_message(message.role, message.content)
        if event is not None:
            content.append(event)

    details = {
        "category": data.category,
        "source": data.source,
    }
    if data.generator is not None:
        details["generator"] = data.generator
    if data.thinking is not None:
        details["thinking"] = data.thinking
    if data.patch:
        details["patch"] = data.patch

    return Trajectory(id=data.id, content=content, details=details)


if __name__ == "__main__":
    for line in sys.stdin:
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        print(standardized_data.model_dump_json())
