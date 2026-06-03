"""Transitional raw-to-ATIF scaffold shared by dataset wrappers.

This is not a native raw→ATIF implementation. It intentionally delegates to each
existing ``raw_to_standardized.py`` converter and then adapts the ADP trajectory
to ATIF so every dataset can expose ATIF samples during the unification work.
As a result, generated ATIF records inherit the ADP conversion's normalization,
filtering, and field-mapping choices; dataset-specific raw→ATIF converters may
replace this wrapper when higher-fidelity ATIF extraction is needed.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schema.atif import adp_trajectory_to_atif
from schema.trajectory import Trajectory


def dataset_dir_from_script(script_file: str) -> Path:
    return Path(script_file).resolve().parent


def main(script_file: str) -> None:
    dataset_dir = dataset_dir_from_script(script_file)
    raw_to_standardized = dataset_dir / "raw_to_standardized.py"
    if not raw_to_standardized.exists():
        raise FileNotFoundError(f"raw_to_standardized.py not found next to {script_file}")

    repo_root = dataset_dir.parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
    raw_input = sys.stdin.read()
    process = subprocess.run(
        [sys.executable, str(raw_to_standardized)],
        input=raw_input,
        text=True,
        capture_output=True,
        cwd=repo_root,
        env=env,
        check=False,
    )
    if process.returncode != 0:
        sys.stderr.write(process.stderr)
        raise SystemExit(process.returncode)
    if process.stderr:
        sys.stderr.write(process.stderr)

    for line in process.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        trajectory = Trajectory(**json.loads(line))
        atif_trajectory = adp_trajectory_to_atif(trajectory)
        print(atif_trajectory.model_dump_json(exclude_none=True))
