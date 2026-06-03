import hashlib
import json
import re
import sys

from schema_raw import SchemaRaw

from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.tool_call_links import create_trajectory_with_tool_call_links
from schema.trajectory import Trajectory

OBSERVATION_PREFIX = "Observation:"
_BASH_BLOCK_RE = re.compile(r"```bash\s*\n(.*?)\n?```", re.DOTALL | re.IGNORECASE)
_THOUGHT_PREFIX_RE = re.compile(r"^THOUGHT:\s*", re.IGNORECASE)


def strip_thought_prefix(content: str) -> str:
    return _THOUGHT_PREFIX_RE.sub("", content.strip()).strip()


def normalize_observation(content: str) -> str:
    if content.startswith(OBSERVATION_PREFIX):
        return content[len(OBSERVATION_PREFIX) :].lstrip()
    return content


def trajectory_id(data: SchemaRaw) -> str:
    serialized_messages = json.dumps(
        [message.model_dump() for message in data.messages],
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha1(serialized_messages.encode("utf-8")).hexdigest()[:12]
    return f"{data.instance_id}-{digest}"


def convert_user_message(content: str) -> TextObservation:
    if content.startswith(OBSERVATION_PREFIX):
        return TextObservation(content=normalize_observation(content), source="environment")
    return TextObservation(content=content, source="user")


def convert_assistant_message(content: str) -> CodeAction | MessageAction:
    bash_matches = list(_BASH_BLOCK_RE.finditer(content))
    if not bash_matches:
        return MessageAction(content=content)

    match = bash_matches[-1]
    description = strip_thought_prefix(content[: match.start()])
    command = match.group(1).strip()
    return CodeAction(language="bash", content=command, description=description or None)


def convert_step(step) -> list:
    if step.role == "system":
        return []
    if step.role == "user":
        return [convert_user_message(step.content)]
    if step.role == "assistant":
        return [convert_assistant_message(step.content)]
    print(f"Unknown role: {step.role}", file=sys.stderr)
    return []


def process_data(data: SchemaRaw) -> Trajectory | None:
    content = []
    for step in data.messages:
        content.extend(convert_step(step))

    if not content:
        return None

    return create_trajectory_with_tool_call_links(
        id=trajectory_id(data),
        content=content,
        details={
            "instance_id": data.instance_id,
            "repo": data.repo,
            "trajectory_format": data.trajectory_format,
            "exit_status": data.exit_status,
            "duration_sec": data.duration_sec,
        },
    )


if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        standardized_data = process_data(data)
        if standardized_data:
            print(standardized_data.model_dump_json())
