#!/usr/bin/env python3
"""Convert CognitiveKernel-Pro-SFT records to ADP standardized trajectories."""

import json
import re
import sys

from schema_raw import SchemaRaw

from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory

THOUGHT_CODE_RE = re.compile(
    r"^\s*Thought:\s*(?P<thought>.*?)\s*\nCode:\s*```(?:python)?\s*\n(?P<code>.*?)\n```\s*$",
    re.DOTALL,
)


def convert_assistant_message(content: str) -> CodeAction | MessageAction:
    match = THOUGHT_CODE_RE.match(content)
    if not match:
        return MessageAction(content=content)

    return CodeAction(
        language="python",
        content=match.group("code").strip(),
        description=match.group("thought").strip(),
    )


def process_record(record: SchemaRaw) -> Trajectory:
    system_messages = [message.content for message in record.messages if message.role == "system"]
    user_messages = [message.content for message in record.messages if message.role == "user"]
    assistant_messages = [
        message.content for message in record.messages if message.role == "assistant"
    ]

    if not user_messages:
        raise ValueError(f"Record {record.id} has no user message")
    if not assistant_messages:
        raise ValueError(f"Record {record.id} has no assistant message")

    prompt_parts = []
    if system_messages:
        prompt_parts.append(
            "## CognitiveKernel System Instructions\n" + "\n\n".join(system_messages)
        )
    prompt_parts.append("## CognitiveKernel Task State\n" + "\n\n".join(user_messages))

    content = [TextObservation(content="\n\n".join(prompt_parts), source="user")]
    content.extend(convert_assistant_message(message) for message in assistant_messages)

    return Trajectory(
        id=record.id,
        content=content,
        details={
            "source": "CognitiveKernel/CognitiveKernel-Pro-SFT",
            "source_file": record.source_file,
            "source_index": str(record.source_index),
        },
    )


if __name__ == "__main__":
    for line in sys.stdin:
        raw_record = SchemaRaw(**json.loads(line))
        print(process_record(raw_record).model_dump_json())
