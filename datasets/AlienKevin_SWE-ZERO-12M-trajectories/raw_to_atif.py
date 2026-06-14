from __future__ import annotations

import hashlib
import json
import re
import sys

from schema_raw import SchemaRaw

from schema.atif import (
    Agent,
    ATIFObservation,
    ATIFTrajectory,
    ObservationResult,
    Step,
    ToolCall,
)

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


def assistant_step(content: str, step_id: int, tool_call_id: str) -> Step:
    bash_matches = list(_BASH_BLOCK_RE.finditer(content))
    if not bash_matches:
        return Step(step_id=step_id, source="agent", message=content)

    match = bash_matches[-1]
    message = strip_thought_prefix(content[: match.start()])
    command = match.group(1).strip()
    return Step(
        step_id=step_id,
        source="agent",
        message=message,
        tool_calls=[
            ToolCall(
                tool_call_id=tool_call_id,
                function_name="bash",
                arguments={"command": command},
            )
        ],
    )


def attach_observation(step: Step, content: str) -> None:
    source_call_id = None
    if step.tool_calls:
        source_call_id = step.tool_calls[-1].tool_call_id
    step.observation = ATIFObservation(
        results=[
            ObservationResult(
                source_call_id=source_call_id,
                content=normalize_observation(content),
            )
        ]
    )


def process_data(data: SchemaRaw) -> ATIFTrajectory | None:
    steps: list[Step] = []
    tool_call_index = 1

    for message in data.messages:
        content = message.content
        if message.role == "system":
            continue
        if message.role == "user":
            if content.startswith(OBSERVATION_PREFIX) and steps and steps[-1].source == "agent":
                attach_observation(steps[-1], content)
            else:
                steps.append(Step(step_id=len(steps) + 1, source="user", message=content))
        elif message.role == "assistant":
            step = assistant_step(
                content,
                step_id=len(steps) + 1,
                tool_call_id=f"call_{tool_call_index:06d}",
            )
            if step.tool_calls:
                tool_call_index += len(step.tool_calls)
            steps.append(step)
        else:
            print(f"Unknown role: {message.role}", file=sys.stderr)

    if not steps:
        return None

    return ATIFTrajectory(
        trajectory_id=trajectory_id(data),
        session_id=data.instance_id,
        agent=Agent(
            name="mini-swe-agent",
            version=str(data.trajectory_format),
        ),
        steps=steps,
        extra={
            "instance_id": data.instance_id,
            "repo": data.repo,
            "trajectory_format": data.trajectory_format,
            "exit_status": data.exit_status,
            "duration_sec": data.duration_sec,
        },
    )


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)
        trajectory = process_data(data)
        if trajectory:
            print(trajectory.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main()
