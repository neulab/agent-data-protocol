from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_SOURCE = Path(__file__).resolve().parents[2] / "enterprise_arena_gold.json"


def load_records(source: Path) -> list[dict[str, Any]]:
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected {source} to contain a JSON list")
    return [record for record in data if isinstance(record, dict)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract EnterpriseLab raw records as JSONL")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to enterprise_arena_gold.json",
    )
    args = parser.parse_args()

    for record in load_records(args.source):
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
