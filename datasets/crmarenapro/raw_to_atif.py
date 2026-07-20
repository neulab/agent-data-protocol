#!/usr/bin/env python3
"""Convert CRMArena-Pro ReAct rollouts to ATIF trajectories.

Unlike the original CRMArena (OpenAI tool-calling messages), CRMArena-Pro
(``--org_type b2b|b2c``) only supports the ReAct ``ChatAgent``, whose trajectory
is a plain text message list:

* ``system`` -> ATIF ``system`` step (the ReAct system prompt).
* ``user`` (first) -> ATIF ``user`` step (the task query). o1-style models merge
  the system prompt into this first user message; it is still mapped to a user
  step.
* ``assistant`` -> ATIF ``agent`` step. The ``<thought>...</thought>`` text
  becomes the step message; a ``<execute>...</execute>`` (SOQL/SOSL/command) or
  ``<respond>...</respond>`` (answer to the user) action becomes a ``ToolCall``.
* ``user`` prefixed ``Salesforce instance output:`` -> the ATIF observation for
  the preceding ``execute`` tool call.
* any other ``user`` message (interactive mode) -> a simulated user turn, mapped
  to a ``user`` step.

Tool calls keep the CRMArena-Pro ``execute``/``respond`` names declared in
``metadata.json``; ``atif_to_std.py`` applies the shared normalization while
leaving these custom tools intact.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from schema.atif import (
    ATIF_SCHEMA_VERSION,
    Agent,
    ATIFObservation,
    ATIFTrajectory,
    ObservationResult,
    Step,
    ToolCall,
)

THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL)
EXECUTE_RE = re.compile(r"<execute>(.*?)</execute>", re.DOTALL)
RESPOND_RE = re.compile(r"<respond>(.*?)</respond>", re.DOTALL)
SF_OUTPUT_PREFIX = "Salesforce instance output:"


def _parse_assistant(content: str, call_id: str) -> tuple[str, list[ToolCall]]:
    """Split an assistant message into (thought message, tool calls)."""
    thought_match = THOUGHT_RE.search(content)
    thought = thought_match.group(1).strip() if thought_match else ""

    execute_match = EXECUTE_RE.search(content)
    respond_match = RESPOND_RE.search(content)

    if execute_match:
        tool_call = ToolCall(
            tool_call_id=call_id,
            function_name="execute",
            arguments={"code": execute_match.group(1).strip()},
            extra={"raw_format": "react_xml"},
        )
        return thought, [tool_call]
    if respond_match:
        tool_call = ToolCall(
            tool_call_id=call_id,
            function_name="respond",
            arguments={"content": respond_match.group(1).strip()},
            extra={"raw_format": "react_xml"},
        )
        return thought, [tool_call]

    # No parseable action (e.g. a malformed turn that triggered a retry rule):
    # keep the full text as the agent message so nothing is lost.
    return (thought or content.strip()), []


def _is_execute_observation(step: Step, content: str) -> bool:
    return (
        step.source == "agent"
        and bool(step.tool_calls)
        and step.tool_calls[-1].function_name == "execute"
        and content.lstrip().startswith(SF_OUTPUT_PREFIX)
    )


def convert_raw_to_atif(record: dict[str, Any]) -> ATIFTrajectory:
    messages = record.get("messages") or []
    steps: list[Step] = []
    call_index = 0

    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""

        if role == "system":
            steps.append(Step(step_id=len(steps) + 1, source="system", message=content.strip()))
        elif role == "user":
            if steps and _is_execute_observation(steps[-1], content):
                obs_text = content.split(SF_OUTPUT_PREFIX, 1)[1].strip()
                source_call_id = steps[-1].tool_calls[-1].tool_call_id
                steps[-1].observation = ATIFObservation(
                    results=[ObservationResult(source_call_id=source_call_id, content=obs_text)]
                )
            else:
                steps.append(Step(step_id=len(steps) + 1, source="user", message=content.strip()))
        elif role == "assistant":
            call_index += 1
            msg, tool_calls = _parse_assistant(content, f"call_{call_index}")
            steps.append(
                Step(
                    step_id=len(steps) + 1,
                    source="agent",
                    message=msg,
                    tool_calls=tool_calls or None,
                )
            )
        # Unknown roles are skipped.

    if not steps:
        steps = [Step(step_id=1, source="user", message="")]

    reward = record.get("reward")
    extra_raw = {key: value for key, value in record.items() if key != "messages"}
    trajectory_id = record.get("id")

    return ATIFTrajectory(
        schema_version=ATIF_SCHEMA_VERSION,
        session_id=trajectory_id,
        trajectory_id=trajectory_id,
        agent=Agent(name="crmarenapro", version="raw"),
        steps=steps,
        final_metrics={"reward": reward} if isinstance(reward, (int, float, bool)) else None,
        extra={"raw": extra_raw, "source_dataset": "crmarenapro"},
    )


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        trajectory = convert_raw_to_atif(record)
        print(trajectory.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main()
