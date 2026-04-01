#!/usr/bin/env python3
"""Extract raw data from the AgentNet dataset on HuggingFace.

Downloads JSONL trajectory files from xlangai/AgentNet, validates with SchemaRaw,
prints quality score distribution stats, and outputs filtered trajectories.

Usage:
    # Basic extraction (JSONL only, no images):
    python datasets/agentnet/extract_raw.py

    # With quality filtering:
    python datasets/agentnet/extract_raw.py --min-alignment=7

    # With image download:
    python datasets/agentnet/extract_raw.py --output-dir=screenshots --download-images

    # Stats only (no output):
    python datasets/agentnet/extract_raw.py --stats-only
"""

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

from schema_raw import SchemaRaw

REPO_ID = "xlangai/AgentNet"

# JSONL files in the HuggingFace repo
JSONL_FILES = {
    "ubuntu": "agentnet_ubuntu_5k.jsonl",
    "win_mac": "agentnet_win_mac_18k.jsonl",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract raw data from AgentNet dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for screenshots (default: ./screenshots)",
    )
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Download and extract image archives from HuggingFace",
    )
    parser.add_argument(
        "--min-alignment",
        type=int,
        default=None,
        help="Minimum alignment_score to include (0-10, default: no filter)",
    )
    parser.add_argument(
        "--completed-only",
        action="store_true",
        default=False,
        help="Only include trajectories with task_completed=True",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print quality stats and exit without outputting data",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of trajectories to output (0=all)",
    )
    parser.add_argument(
        "--platforms",
        type=str,
        default="all",
        help="Comma-separated platforms to include: ubuntu,win_mac,all (default: all)",
    )
    return parser.parse_args()


def download_jsonl_files(platforms: list[str]) -> dict[str, Path]:
    """Download JSONL files from HuggingFace, returning {platform: local_path}."""
    paths = {}
    for platform in platforms:
        filename = JSONL_FILES[platform]
        print(f"Downloading {filename}...", file=sys.stderr)
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
        )
        paths[platform] = Path(local_path)
        print(f"  Downloaded to {local_path}", file=sys.stderr)
    return paths


def print_stats(trajectories: list[dict]) -> None:
    """Print quality score distribution stats to stderr."""
    total = len(trajectories)
    print(f"\n{'=' * 50}", file=sys.stderr)
    print(f"AgentNet Quality Stats ({total} trajectories)", file=sys.stderr)
    print(f"{'=' * 50}", file=sys.stderr)

    # task_completed distribution
    completed_counts = Counter(t.get("task_completed") for t in trajectories)
    print(f"\ntask_completed:", file=sys.stderr)
    for val, count in sorted(completed_counts.items(), key=lambda x: str(x[0])):
        pct = count / total * 100
        print(f"  {val}: {count} ({pct:.1f}%)", file=sys.stderr)

    # Score distributions
    for field in ["alignment_score", "efficiency_score", "task_difficulty"]:
        values = [t.get(field) for t in trajectories if t.get(field) is not None]
        if not values:
            print(f"\n{field}: no data", file=sys.stderr)
            continue
        score_counts = Counter(values)
        print(f"\n{field} (n={len(values)}):", file=sys.stderr)
        for score in range(11):
            count = score_counts.get(score, 0)
            pct = count / len(values) * 100
            bar = "#" * int(pct / 2)
            print(f"  {score:2d}: {count:5d} ({pct:5.1f}%) {bar}", file=sys.stderr)

    # Cumulative: how many trajectories pass various alignment thresholds
    alignment_values = [
        t.get("alignment_score") for t in trajectories if t.get("alignment_score") is not None
    ]
    if alignment_values:
        completed_trajs = [t for t in trajectories if t.get("task_completed")]
        print(f"\nCumulative (completed_only + alignment >= N):", file=sys.stderr)
        for threshold in range(5, 11):
            passing = sum(
                1
                for t in completed_trajs
                if t.get("alignment_score") is not None
                and t["alignment_score"] >= threshold
            )
            pct = passing / total * 100
            print(
                f"  completed + alignment >= {threshold}: {passing:5d} ({pct:.1f}%)",
                file=sys.stderr,
            )

    # Trajectory length stats
    lengths = [len(t.get("traj", [])) for t in trajectories]
    if lengths:
        print(f"\nTrajectory length:", file=sys.stderr)
        print(f"  min: {min(lengths)}, max: {max(lengths)}, "
              f"mean: {sum(lengths) / len(lengths):.1f}, "
              f"median: {sorted(lengths)[len(lengths) // 2]}", file=sys.stderr)

    print(f"{'=' * 50}\n", file=sys.stderr)


def download_images(output_dir: Path, platforms: list[str]) -> None:
    """Download and extract image archives from HuggingFace.

    AgentNet images are stored in split zip archives on HuggingFace.
    This downloads and extracts them to the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Image archive mapping: platform -> HF directory with zip files
    image_dirs = {
        "ubuntu": "ubuntu_images",
        "win_mac": "win_mac_images",
    }

    for platform in platforms:
        img_dir = image_dirs[platform]
        print(f"Downloading images for {platform}...", file=sys.stderr)

        # Download the main zip file
        # AgentNet uses split archives: .z01, .z02, ... .zip
        # The .zip file is the final part that can list all contents
        try:
            zip_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=f"{img_dir}/{img_dir}.zip",
            )
            print(f"  Extracting {zip_path} to {output_dir}...", file=sys.stderr)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(output_dir)
            print(f"  Done extracting {platform} images.", file=sys.stderr)
        except Exception as e:
            print(
                f"  WARNING: Could not download/extract {platform} images: {e}",
                file=sys.stderr,
            )
            print(
                f"  Split zip archives may require manual download. "
                f"See README.md for instructions.",
                file=sys.stderr,
            )


def main() -> None:
    """Extract AgentNet trajectories from HuggingFace."""
    args = parse_args()

    # Determine platforms
    if args.platforms == "all":
        platforms = list(JSONL_FILES.keys())
    else:
        platforms = [p.strip() for p in args.platforms.split(",")]
        for p in platforms:
            if p not in JSONL_FILES:
                print(
                    f"ERROR: Unknown platform '{p}'. Choose from: {', '.join(JSONL_FILES.keys())}",
                    file=sys.stderr,
                )
                sys.exit(1)

    # Download JSONL files
    jsonl_paths = download_jsonl_files(platforms)

    # Read all trajectories
    all_trajectories: list[dict] = []
    validation_errors = 0
    for platform, path in jsonl_paths.items():
        print(f"Reading {path.name}...", file=sys.stderr)
        with open(path, "r") as f:
            for line in tqdm(f, desc=f"Parsing {platform}", file=sys.stderr):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    # Validate with Pydantic schema
                    SchemaRaw(**raw)
                    raw["_platform"] = platform
                    all_trajectories.append(raw)
                except Exception as e:
                    validation_errors += 1
                    if validation_errors <= 5:
                        print(
                            f"  Validation error: {e}",
                            file=sys.stderr,
                        )

    if validation_errors > 0:
        print(
            f"Total validation errors: {validation_errors}",
            file=sys.stderr,
        )

    # Print stats
    print_stats(all_trajectories)

    if args.stats_only:
        return

    # Apply filters
    filtered = all_trajectories
    if args.completed_only:
        filtered = [t for t in filtered if t.get("task_completed")]
        print(
            f"After completed_only filter: {len(filtered)}/{len(all_trajectories)}",
            file=sys.stderr,
        )
    if args.min_alignment is not None:
        filtered = [
            t
            for t in filtered
            if t.get("alignment_score") is not None
            and t["alignment_score"] >= args.min_alignment
        ]
        print(
            f"After alignment >= {args.min_alignment} filter: {len(filtered)}/{len(all_trajectories)}",
            file=sys.stderr,
        )

    # Apply limit
    if args.limit > 0:
        filtered = filtered[: args.limit]

    # Download images if requested
    if args.download_images:
        script_dir = Path(__file__).parent
        output_dir = Path(args.output_dir) if args.output_dir else script_dir / "screenshots"
        download_images(output_dir, platforms)

    # Output
    print(f"Outputting {len(filtered)} trajectories...", file=sys.stderr)
    for traj in filtered:
        # Remove internal _platform field before output
        platform = traj.pop("_platform", None)
        print(json.dumps(traj))


if __name__ == "__main__":
    main()
