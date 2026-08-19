from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any, Iterator

signal.signal(signal.SIGPIPE, signal.SIG_DFL)

# By default, look for CRMArena rollout logs placed at the repository root. Each
# file must be a JSON list of records produced by the CRMArena tool-calling
# agent, where every record has a ``traj`` field holding an OpenAI-style message
# list (system/user/assistant tool_calls/tool observations/respond).
DEFAULT_SOURCE = Path(__file__).resolve().parents[2] / "crmarena_rollouts.json"


def load_records(source: Path) -> list[dict[str, Any]]:
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected {source} to contain a JSON list of rollout records")
    return [record for record in data if isinstance(record, dict)]


def to_raw_record(record: dict[str, Any]) -> dict[str, Any] | None:
    messages = record.get("traj") or record.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    task_type = record.get("task_type")
    task_id = record.get("task_id")
    trajectory_id = "_".join(
        str(part) for part in ("crmarena", task_type, task_id) if part is not None
    )
    raw: dict[str, Any] = {"id": trajectory_id, "messages": messages}
    for field in ("task_id", "task_type", "gt_answer", "reward", "agent_info"):
        if field in record:
            raw[field] = record[field]
    return raw


def iter_raw_records(sources: list[Path]) -> Iterator[dict[str, Any]]:
    for source in sources:
        for record in load_records(source):
            raw = to_raw_record(record)
            if raw is not None:
                yield raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CRMArena rollout logs as raw JSONL")
    parser.add_argument(
        "--source",
        type=Path,
        nargs="+",
        default=[DEFAULT_SOURCE],
        help="One or more CRMArena result JSON files (lists of records with a 'traj' field)",
    )
    args = parser.parse_args()

    try:
        for raw in iter_raw_records(args.source):
            print(json.dumps(raw, ensure_ascii=False))
    except BrokenPipeError:
        sys.exit(0)


if __name__ == "__main__":
    main()
