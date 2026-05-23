import json
import os
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

REPO_ID = "SALT-NLP/SWE-chat"
SPLIT = "train"


def load_hf_dataset(config: str, streaming: bool = False, token: str | None = None):
    from datasets import load_dataset

    try:
        return load_dataset(REPO_ID, config, split=SPLIT, streaming=streaming, token=token)
    except TypeError:
        return load_dataset(REPO_ID, config, split=SPLIT, streaming=streaming, use_auth_token=token)


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    return str(value)


def emit(item: dict[str, Any]) -> None:
    try:
        print(json.dumps(json_safe(item), ensure_ascii=False))
    except BrokenPipeError:
        os._exit(0)


def load_session_metadata(token: str | None) -> dict[str, dict[str, Any]]:
    try:
        sessions = load_hf_dataset("sessions", streaming=False, token=token)
    except Exception:
        return {}
    metadata = {}
    for row in sessions:
        session_id = row.get("session_id")
        if session_id:
            metadata[session_id] = dict(row)
    return metadata


def compact_turn(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if value is not None}


def build_session(
    session_id: str,
    turns: list[dict[str, Any]],
    session_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    session_metadata = dict(session_metadata or {})
    first_turn = turns[0] if turns else {}
    item = {
        "session_id": session_id,
        "repo_id": session_metadata.get("repo_id") or first_turn.get("repo_id"),
        "checkpoint_pk": first_turn.get("checkpoint_pk")
        or session_metadata.get("canonical_checkpoint_pk"),
        "user_id": session_metadata.get("user_id") or first_turn.get("user_id"),
        "agent": session_metadata.get("agent") or first_turn.get("agent"),
        "strategy": session_metadata.get("strategy") or first_turn.get("strategy"),
        "branch": session_metadata.get("branch"),
        "created_at": session_metadata.get("created_at") or first_turn.get("timestamp"),
        "transcript_path": session_metadata.get("transcript_path"),
        "tool_call_count": session_metadata.get("tool_call_count"),
        "turn_count": session_metadata.get("turn_count"),
        "prompt_count": session_metadata.get("prompt_count"),
        "agent_percentage": session_metadata.get("agent_percentage"),
        "user_persona": session_metadata.get("user_persona"),
        "session_success": session_metadata.get("session_success"),
        "turns": [compact_turn(turn) for turn in turns],
    }
    return {key: value for key, value in item.items() if value is not None}


def iter_grouped_sessions(
    rows: Iterable[dict[str, Any]],
    session_metadata: dict[str, dict[str, Any]],
    max_turns_per_session: int | None = None,
) -> Iterable[dict[str, Any]]:
    current_session_id = None
    current_turns: list[dict[str, Any]] = []

    for row in rows:
        session_id = row.get("session_id")
        if not session_id:
            continue
        if current_session_id is not None and session_id != current_session_id:
            yield build_session(
                current_session_id, current_turns, session_metadata.get(current_session_id)
            )
            current_turns = []
        current_session_id = session_id
        if max_turns_per_session is None or len(current_turns) < max_turns_per_session:
            current_turns.append(dict(row))

    if current_session_id is not None:
        yield build_session(
            current_session_id, current_turns, session_metadata.get(current_session_id)
        )


def main() -> None:
    token = os.environ.get("HF_TOKEN") or None
    max_turns = os.environ.get("SWE_CHAT_MAX_TURNS_PER_SESSION")
    max_turns_per_session = int(max_turns) if max_turns else None
    session_metadata = load_session_metadata(token)
    conversations = load_hf_dataset("conversations", streaming=True, token=token)
    for item in iter_grouped_sessions(conversations, session_metadata, max_turns_per_session):
        emit(item)


if __name__ == "__main__":
    main()
