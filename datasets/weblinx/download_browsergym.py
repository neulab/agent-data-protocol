#!/usr/bin/env python3
"""Download pre-computed axtrees from McGill-NLP/weblinx-browsergym.

Downloads axtree and extra_element_properties files from HuggingFace
for use during weblinx standardization (raw_to_standardized.py).

Uses per-file downloads via hf_hub_download (not snapshot_download)
because the repo has 498k+ files and HuggingFace tree listing times out.

Usage:
    python datasets/weblinx/download_browsergym.py --output-dir /data/datasets/weblinx/browsergym-data
    python datasets/weblinx/download_browsergym.py --split train --output-dir ./browsergym-data
"""

import argparse
import json
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ID = "McGill-NLP/weblinx-browsergym"
METADATA_FILE = "metadata.json"


def log(msg: str) -> None:
    """Print to stderr and flush immediately."""
    print(msg, file=sys.stderr, flush=True)


def download_metadata(output_dir: Path) -> dict:
    """Download and cache metadata.json from weblinx-browsergym.

    Args:
        output_dir: Directory to store downloaded files.

    Returns:
        Parsed metadata dictionary.
    """
    metadata_path = output_dir / METADATA_FILE
    if not metadata_path.exists():
        log(f"Downloading {METADATA_FILE}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=METADATA_FILE,
            repo_type="dataset",
            local_dir=output_dir,
        )
        log(f"  Saved to {metadata_path}")
    else:
        log(f"Using cached {metadata_path}")

    log("Loading metadata.json...")
    with open(metadata_path) as f:
        metadata = json.load(f)
    log(f"  Loaded ({len(metadata)} splits)")
    return metadata


def collect_files_to_download(metadata: dict, split: str) -> list[str]:
    """Collect all axtree and extra_element_properties file paths from metadata.

    Args:
        metadata: Parsed metadata.json.
        split: Dataset split (e.g. 'train').

    Returns:
        List of file paths relative to repo root (e.g.
        'demonstrations/cptbbef/axtrees/page-2-0.json').
    """
    files = set()
    demos = metadata[split]

    for shortcode, steps in demos.items():
        for step_idx, step in steps.items():
            axtree_path = step.get("axtree_path")
            if axtree_path:
                files.add(f"demonstrations/{axtree_path}")

            props_path = step.get("extra_props_path")
            if props_path:
                files.add(f"demonstrations/{props_path}")

    return sorted(files)


def download_files(
    output_dir: Path,
    files: list[str],
) -> tuple[int, int]:
    """Download individual files from HuggingFace.

    Args:
        output_dir: Directory to store downloaded files.
        files: List of file paths relative to repo root.

    Returns:
        Tuple of (successful_downloads, failed_downloads).
    """
    success = 0
    skipped = 0
    failed = 0
    failed_paths = []

    for filepath in tqdm(files, desc="Downloading", file=sys.stderr, miniters=100):
        target = output_dir / filepath
        if target.exists():
            skipped += 1
            continue

        max_retries = 3
        for attempt in range(max_retries):
            try:
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=filepath,
                    repo_type="dataset",
                    local_dir=output_dir,
                )
                success += 1
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s
                else:
                    failed += 1
                    failed_paths.append(filepath)
                    if failed <= 20:
                        log(f"  Failed ({failed}): {filepath}: {e}")
                    elif failed == 21:
                        log("  (suppressing further failure messages)")

    return success, skipped, failed, failed_paths


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download pre-computed axtrees from weblinx-browsergym.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store downloaded files (will contain demonstrations/).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to download (default: train).",
    )
    args = parser.parse_args()

    log(f"Starting download_browsergym.py")
    log(f"  output_dir: {args.output_dir}")
    log(f"  split: {args.split}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Download metadata
    metadata = download_metadata(args.output_dir)

    # Get shortcodes for requested split
    if args.split not in metadata:
        log(f"Error: split '{args.split}' not found. Available: {list(metadata.keys())}")
        sys.exit(1)

    log(f"Split '{args.split}': {len(metadata[args.split])} demonstrations")

    # Collect files from metadata
    files = collect_files_to_download(metadata, args.split)
    log(f"Files to download: {len(files)}")

    # Download
    success, skipped, failed, failed_paths = download_files(args.output_dir, files)
    log(f"Done: {success} downloaded, {skipped} cached, {failed} failed (total: {success + skipped + failed})")

    if failed > 0:
        log(f"Failed files written to: {args.output_dir / 'failed_downloads.txt'}")
        with open(args.output_dir / "failed_downloads.txt", "w") as f:
            for p in failed_paths:
                f.write(p + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
