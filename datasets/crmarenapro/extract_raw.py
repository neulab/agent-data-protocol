from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any, Iterator

signal.signal(signal.SIGPIPE, signal.SIG_DFL)

# By default, look for CRMArena-Pro rollout logs placed at the repository root.
# Each file must be a JSON list of records produced by the CRMArena ReAct
# ChatAgent (``--agent_strategy react`` with ``--org_type b2b`` or ``b2c``),
# where every record has a ``traj`` field holding a message list of
# system/user/assistant turns. Assistant turns carry ``<thought>`` reasoning
# plus a ``<execute>`` (SOQL/SOSL) or ``<respond>`` (answer to the user) action;
# execute results come back as ``user`` messages prefixed
# ``Salesforce instance output:``.
DEFAULT_SOURCE = Path(__file__).resolve().parents[2] / "crmarenapro_rollouts.json"


def infer_org_type(source: Path) -> str | None:
    """CRMArena-Pro results live under a b2b/b2c directory or file name."""
    lowered = source.as_posix().lower()
    if "b2c" in lowered:
        return "b2c"
    if "b2b" in lowered:
        return "b2b"
    return None


def infer_interactive(source: Path) -> bool | None:
    """Interactive (multi-turn) rollouts are tagged in the path/file name."""
    lowered = source.as_posix().lower()
    if "interactive-true" in lowered:
        return True
    if "interactive-false" in lowered:
        return False
    if "interactive" in lowered:
        return True
    return None


def load_records(source: Path) -> list[dict[str, Any]]:
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected {source} to contain a JSON list of rollout records")
    return [record for record in data if isinstance(record, dict)]


def to_raw_record(
    record: dict[str, Any], org_type: str | None, interactive: bool | None
) -> dict[str, Any] | None:
    messages = record.get("traj") or record.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    task_type = record.get("task_type")
    task_id = record.get("task_id")
    trajectory_id = "_".join(
        str(part) for part in ("crmarenapro", org_type, task_type, task_id) if part is not None
    )
    raw: dict[str, Any] = {"id": trajectory_id, "messages": messages}
    if org_type is not None:
        raw["org_type"] = org_type
    if interactive is not None:
        raw["interactive"] = interactive
    for field in ("task_id", "task_type", "gt_answer", "reward", "agent_info"):
        if field in record:
            raw[field] = record[field]
    return raw


def iter_raw_records(sources: list[Path]) -> Iterator[dict[str, Any]]:
    for source in sources:
        org_type = infer_org_type(source)
        interactive = infer_interactive(source)
        for record in load_records(source):
            raw = to_raw_record(record, org_type, interactive)
            if raw is not None:
                yield raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract CRMArena-Pro ReAct rollout logs as raw JSONL"
    )
    parser.add_argument(
        "--source",
        type=Path,
        nargs="+",
        default=[DEFAULT_SOURCE],
        help="One or more CRMArena-Pro result JSON files (lists of records with a 'traj' field)",
    )
    args = parser.parse_args()

    try:
        for raw in iter_raw_records(args.source):
            print(json.dumps(raw, ensure_ascii=False))
    except BrokenPipeError:
        sys.exit(0)


if __name__ == "__main__":
    main()
