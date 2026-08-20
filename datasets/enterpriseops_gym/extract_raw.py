from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DATASET_DIR = Path(__file__).resolve().parent
# Full trajectories produced by generate_trajectories.py are written here.
DEFAULT_SOURCE = DATASET_DIR / "enterpriseops_gym_gold.json"
# Committed reference trajectories used when the full gold file is absent.
FALLBACK_SOURCE = DATASET_DIR / "sample_raw.json"


def load_records(source: Path) -> list[dict[str, Any]]:
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected {source} to contain a JSON list")
    return [record for record in data if isinstance(record, dict)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract EnterpriseOps-Gym raw records as JSONL")
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help=(
            "Path to EnterpriseOps-Gym trajectories JSON. Defaults to "
            "enterpriseops_gym_gold.json when present, otherwise falls back to the "
            "committed sample_raw.json."
        ),
    )
    args = parser.parse_args()

    source = args.source
    if source is None:
        source = DEFAULT_SOURCE if DEFAULT_SOURCE.exists() else FALLBACK_SOURCE

    for record in load_records(source):
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
