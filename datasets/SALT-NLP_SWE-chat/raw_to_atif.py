from __future__ import annotations

import json
import sys
from typing import Any

from scripts.raw_to_atif_common import trajectory_from_record


def flattened_session(record: dict[str, Any]) -> dict[str, Any]:
    session = record.get("session")
    turns = record.get("turns", [])
    if not isinstance(session, dict):
        session = {}
    first_turn = turns[0] if turns and isinstance(turns[0], dict) else {}
    return {
        "session_id": record.get("session_id")
        or session.get("session_id")
        or first_turn.get("session_id"),
        "repo_id": session.get("repo_id") or first_turn.get("repo_id"),
        "checkpoint_pk": first_turn.get("checkpoint_pk") or session.get("canonical_checkpoint_pk"),
        "user_id": session.get("user_id") or first_turn.get("user_id"),
        "agent": session.get("agent") or first_turn.get("agent"),
        "strategy": session.get("strategy") or first_turn.get("strategy"),
        "branch": session.get("branch"),
        "created_at": session.get("created_at") or first_turn.get("timestamp"),
        "transcript_path": session.get("transcript_path"),
        "tool_call_count": session.get("tool_call_count"),
        "turn_count": session.get("turn_count"),
        "prompt_count": session.get("prompt_count"),
        "agent_percentage": session.get("agent_percentage"),
        "user_persona": session.get("user_persona"),
        "session_success": session.get("session_success"),
        "turns": turns,
    }


def main(script_file: str) -> None:
    dataset_name = script_file.rsplit("/", 2)[-2]
    for index, line in enumerate(sys.stdin):
        if not line.strip():
            continue
        raw_record = json.loads(line)
        trajectory = trajectory_from_record(flattened_session(raw_record), index, dataset_name)
        if trajectory.extra is not None:
            trajectory.extra["raw"] = raw_record
        print(trajectory.model_dump_json(exclude_none=True))


if __name__ == "__main__":
    main(__file__)
