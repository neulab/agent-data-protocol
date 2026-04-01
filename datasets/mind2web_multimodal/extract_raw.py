#!/usr/bin/env python3
"""Extract raw data from the Multimodal-Mind2Web dataset on HuggingFace.

Downloads per-action rows from osunlp/Multimodal-Mind2Web, groups them into
per-trajectory JSONL, optionally saves original screenshots, and outputs
to stdout.

Usage:
    # Basic extraction:
    python datasets/mind2web_multimodal/extract_raw.py

    # With original screenshot download:
    python datasets/mind2web_multimodal/extract_raw.py --output-dir=datasets/mind2web --download-images

    # Stats only:
    python datasets/mind2web_multimodal/extract_raw.py --stats-only

    # Limit to 5 trajectories for samples:
    python datasets/mind2web_multimodal/extract_raw.py --limit=5
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from schema_raw import ActionStep, Operation, SchemaRaw

REPO_ID = "osunlp/Multimodal-Mind2Web"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract raw data from Multimodal-Mind2Web dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for screenshots (default: current dir)",
    )
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Save original screenshots to disk",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="HuggingFace split to use (default: train)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print distribution stats and exit without outputting data",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max trajectories to output (0=all)",
    )
    return parser.parse_args()


def parse_operation(operation_str: str) -> dict:
    """Parse JSON-encoded operation string into dict."""
    return json.loads(operation_str)


def extract_backend_node_id(pos_candidates: list[str]) -> str | None:
    """Extract backend_node_id from first positive candidate.

    Returns None if pos_candidates is empty (5.8% of actions — mostly
    SVG clicks and ambiguous elements).
    """
    if not pos_candidates:
        return None
    parsed = json.loads(pos_candidates[0])
    return parsed.get("backend_node_id")


def save_screenshot(image, output_dir: Path, annotation_id: str, action_index: int) -> str:
    """Save original screenshot to disk, return relative path."""
    screenshot_dir = output_dir / "screenshots" / "original" / annotation_id
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{action_index}.jpg"
    filepath = screenshot_dir / filename
    image.save(filepath, format="JPEG")
    return str(filepath)


def process_trajectory(
    rows: list[dict],
    output_dir: Path | None,
    download_images: bool,
) -> SchemaRaw:
    """Convert a group of per-action rows into a single trajectory.

    Args:
        rows: Per-action rows from HF dataset, sorted by target_action_index.
        output_dir: Directory for saving screenshots.
        download_images: Whether to save original screenshots.

    Returns:
        Validated SchemaRaw trajectory object.
    """
    first = rows[0]

    actions = []
    for row in rows:
        operation = parse_operation(row["operation"])
        backend_node_id = extract_backend_node_id(row["pos_candidates"])

        screenshot_path = None
        if download_images and output_dir and row.get("screenshot"):
            action_index = int(row["target_action_index"])
            screenshot_path = save_screenshot(
                row["screenshot"],
                output_dir,
                first["annotation_id"],
                action_index,
            )

        actions.append(
            ActionStep(
                action_uid=f"{first['annotation_id']}_{row['target_action_index']}",
                cleaned_html=row["cleaned_html"],
                operation=Operation(**operation),
                backend_node_id=backend_node_id,
                screenshot_path=screenshot_path,
                action_repr=row["target_action_reprs"],
            )
        )

    return SchemaRaw(
        annotation_id=first["annotation_id"],
        website=first["website"],
        domain=first["domain"],
        subdomain=first["subdomain"],
        confirmed_task=first["confirmed_task"],
        action_reprs=first["action_reprs"],
        actions=actions,
    )


def print_stats(trajectories_meta: list[dict]) -> None:
    """Print distribution stats to stderr."""
    domains = Counter(t["domain"] for t in trajectories_meta)
    websites = Counter(t["website"] for t in trajectories_meta)
    action_counts = [t["num_actions"] for t in trajectories_meta]
    empty_pos = sum(t["empty_pos_count"] for t in trajectories_meta)
    total_actions = sum(t["num_actions"] for t in trajectories_meta)

    print(f"\n=== Mind2Web Extraction Stats ===", file=sys.stderr)
    print(f"Trajectories: {len(trajectories_meta)}", file=sys.stderr)
    print(f"Total actions: {total_actions}", file=sys.stderr)
    print(f"Actions/trajectory: min={min(action_counts)}, max={max(action_counts)}, "
          f"avg={sum(action_counts)/len(action_counts):.1f}", file=sys.stderr)
    print(f"Empty pos_candidates: {empty_pos}/{total_actions} "
          f"({100*empty_pos/total_actions:.1f}%)", file=sys.stderr)
    print(f"\nDomains ({len(domains)}):", file=sys.stderr)
    for domain, count in domains.most_common(10):
        print(f"  {domain}: {count}", file=sys.stderr)
    if len(domains) > 10:
        print(f"  ... and {len(domains)-10} more", file=sys.stderr)
    print(f"\nWebsites: {len(websites)} unique", file=sys.stderr)


def main() -> None:
    """Main extraction pipeline."""
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None

    print(f"Loading {REPO_ID} split={args.split}...", file=sys.stderr)
    ds = load_dataset(REPO_ID, split=args.split, streaming=True)

    # Group consecutive rows by annotation_id
    current_aid = None
    current_rows = []
    trajectory_count = 0
    trajectories_meta = []

    for row in ds:
        aid = row["annotation_id"]

        if aid != current_aid:
            # Emit previous trajectory
            if current_rows:
                traj = process_trajectory(current_rows, output_dir, args.download_images)
                empty_pos = sum(1 for a in traj.actions if a.backend_node_id is None)
                trajectories_meta.append({
                    "domain": traj.domain,
                    "website": traj.website,
                    "num_actions": len(traj.actions),
                    "empty_pos_count": empty_pos,
                })

                if not args.stats_only:
                    print(traj.model_dump_json())

                trajectory_count += 1
                if args.limit and trajectory_count >= args.limit:
                    break

                if trajectory_count % 100 == 0:
                    print(f"Processed {trajectory_count} trajectories...", file=sys.stderr)

            current_aid = aid
            current_rows = []

        current_rows.append(row)

    # Emit last trajectory
    if current_rows and (not args.limit or trajectory_count < args.limit):
        traj = process_trajectory(current_rows, output_dir, args.download_images)
        empty_pos = sum(1 for a in traj.actions if a.backend_node_id is None)
        trajectories_meta.append({
            "domain": traj.domain,
            "website": traj.website,
            "num_actions": len(traj.actions),
            "empty_pos_count": empty_pos,
        })
        if not args.stats_only:
            print(traj.model_dump_json())
        trajectory_count += 1

    print_stats(trajectories_meta)
    print(f"\nDone: {trajectory_count} trajectories output.", file=sys.stderr)


if __name__ == "__main__":
    main()
