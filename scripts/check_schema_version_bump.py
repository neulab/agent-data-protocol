#!/usr/bin/env python3
"""Ensure schema changes are accompanied by an ADP schema version bump."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION_PATH = "schema/version.py"
_VERSION_ASSIGNMENT_RE = re.compile(
    r"^SCHEMA_VERSION(?::[^=]+)?\s*=\s*[\"']([^\"']+)[\"']",
    re.MULTILINE,
)
_SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_EXCLUDED_SCHEMA_FILES = {SCHEMA_VERSION_PATH, "schema/__init__.py"}


def run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def changed_files(base_ref: str, head_ref: str, cwd: Path) -> list[str]:
    result = run_git(["diff", "--name-only", f"{base_ref}...{head_ref}"], cwd)
    if result.returncode != 0:
        raise RuntimeError(
            "Unable to determine changed files with "
            f"'git diff --name-only {base_ref}...{head_ref}'.\n{result.stderr}"
        )
    return [line for line in result.stdout.splitlines() if line]


def is_schema_impacting_file(path: str) -> bool:
    return (
        path.startswith("schema/") and path.endswith(".py") and path not in _EXCLUDED_SCHEMA_FILES
    )


def extract_schema_version(content: str) -> str | None:
    match = _VERSION_ASSIGNMENT_RE.search(content)
    if match is None:
        return None
    return match.group(1)


def parse_semver(version: str) -> tuple[int, int, int]:
    match = _SEMVER_RE.fullmatch(version)
    if match is None:
        raise ValueError(f"{version!r} is not a valid MAJOR.MINOR.PATCH version")
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def file_at_ref(ref: str, path: str, cwd: Path) -> str | None:
    result = run_git(["show", f"{ref}:{path}"], cwd)
    if result.returncode != 0:
        return None
    return result.stdout


def schema_version_at_ref(ref: str, cwd: Path) -> tuple[int, int, int] | None:
    content = file_at_ref(ref, SCHEMA_VERSION_PATH, cwd)
    if content is None:
        return None

    version = extract_schema_version(content)
    if version is None:
        raise ValueError(f"{SCHEMA_VERSION_PATH} at {ref!r} does not define SCHEMA_VERSION")
    return parse_semver(version)


def format_paths(paths: list[str]) -> str:
    return "\n".join(f"  - {path}" for path in paths)


def check_schema_version_bump(base_ref: str, head_ref: str, cwd: Path) -> tuple[bool, str]:
    files = changed_files(base_ref, head_ref, cwd)
    schema_files = [path for path in files if is_schema_impacting_file(path)]
    if not schema_files:
        return True, "No schema-impacting Python files changed."

    try:
        base_version = schema_version_at_ref(base_ref, cwd)
        head_version = schema_version_at_ref(head_ref, cwd)
    except ValueError as exc:
        return False, str(exc)

    if head_version is None:
        return False, f"{SCHEMA_VERSION_PATH} is missing at {head_ref!r}."

    if base_version is None:
        return (
            True,
            "Schema versioning is being introduced for the first time with "
            f"SCHEMA_VERSION={'.'.join(str(part) for part in head_version)}.",
        )

    if head_version <= base_version:
        return (
            False,
            "Schema-impacting files changed, but SCHEMA_VERSION was not increased.\n"
            f"Base version: {'.'.join(str(part) for part in base_version)}\n"
            f"Head version: {'.'.join(str(part) for part in head_version)}\n"
            f"Changed schema files:\n{format_paths(schema_files)}",
        )

    return (
        True,
        "Schema version bump check passed: "
        f"{'.'.join(str(part) for part in base_version)} -> "
        f"{'.'.join(str(part) for part in head_version)}.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", required=True, help="Base git ref to compare from")
    parser.add_argument("--head-ref", default="HEAD", help="Head git ref to compare to")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root where git commands should run. Defaults to current directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ok, message = check_schema_version_bump(
        base_ref=args.base_ref,
        head_ref=args.head_ref,
        cwd=Path(args.repo_root).resolve(),
    )
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
