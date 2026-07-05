"""Check or refresh generated dataset sample fixtures.

The check regenerates ``sample_std.json`` from each dataset's checked-in
``sample_atif.json`` and compares it with the committed fixture.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"


def dataset_dirs(names: list[str]) -> list[Path]:
    if names:
        return [DATASETS_DIR / name for name in names]
    return sorted(path for path in DATASETS_DIR.iterdir() if path.is_dir())


def render_jsonl(items: list[object]) -> str:
    return "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)


def regenerate_sample_std(dataset_dir: Path) -> str:
    sample_atif = dataset_dir / "sample_atif.json"
    converter = dataset_dir / "atif_to_std.py"
    if not sample_atif.exists() or not converter.exists():
        raise FileNotFoundError(f"{dataset_dir.name} is missing sample_atif.json or atif_to_std.py")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    atif_rows = json.loads(sample_atif.read_text())
    result = subprocess.run(
        [sys.executable, str(converter)],
        input=render_jsonl(atif_rows),
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )
    std_rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    return json.dumps(std_rows, indent=2, ensure_ascii=True) + "\n"


def check_dataset(dataset_dir: Path, update: bool) -> tuple[bool, str]:
    sample_std = dataset_dir / "sample_std.json"
    if not sample_std.exists():
        return True, f"skip {dataset_dir.name}: missing sample_std.json"

    regenerated = regenerate_sample_std(dataset_dir)
    current = sample_std.read_text()
    if regenerated == current:
        return True, f"ok {dataset_dir.name}"

    if update:
        sample_std.write_text(regenerated)
        return True, f"updated {dataset_dir.name}"

    diff = "".join(
        difflib.unified_diff(
            current.splitlines(keepends=True),
            regenerated.splitlines(keepends=True),
            fromfile=f"{dataset_dir.name}/sample_std.json",
            tofile=f"{dataset_dir.name}/sample_std.json regenerated",
            n=3,
        )
    )
    return False, f"drift {dataset_dir.name}\n{diff}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset directory name to check. May be repeated. Defaults to all datasets.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Rewrite stale sample_std.json files instead of failing.",
    )
    args = parser.parse_args()

    ok = True
    for dataset_dir in dataset_dirs(args.dataset):
        try:
            dataset_ok, message = check_dataset(dataset_dir, args.update)
        except Exception as exc:
            dataset_ok = False
            message = f"error {dataset_dir.name}: {exc}"
        ok = ok and dataset_ok
        print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
