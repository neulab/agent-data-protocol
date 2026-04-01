#!/usr/bin/env python3
"""Convert Mind2Web raw data to ADP standardized format.

Reads per-trajectory JSONL from stdin (output of extract_raw.py), renders
cleaned_html in BrowserGym to generate axtrees with BIDs, resolves
backend_node_id to BID, generates SoM-annotated screenshots, and outputs
Trajectory objects.

Requires BrowserGym + Chromium (Playwright) for axtree generation and
SoM screenshot rendering.

Usage:
    cat sample_raw.json | python scripts/json_to_jsonl.py | \
        python datasets/mind2web_multimodal/raw_to_standardized.py --filter-som | \
        python scripts/jsonl_to_json.py > sample_std.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from browsergym.utils.obs import overlay_som
from PIL import Image

sys.path.append(".")

from schema.action.api import ApiAction
from schema.observation.image import ImageObservation
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory
from scripts.html_to_axtree import HTMLToAXTree

from schema_raw import SchemaRaw

# Map Mind2Web operation types to api.py function names
OPERATION_MAP = {
    "CLICK": "click",
    "TYPE": "type",
    "SELECT": "select",
}

# Screenshots directory relative to repo root
SCREENSHOTS_DIR = "datasets/mind2web_multimodal/screenshots"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Mind2Web raw data to standardized format"
    )
    parser.add_argument(
        "--filter-som",
        action="store_true",
        default=True,
        help="Filter axtree to interactive elements only (default: True)",
    )
    parser.add_argument(
        "--no-filter-som",
        action="store_true",
        help="Disable SoM filtering (include all elements in axtree)",
    )
    parser.add_argument(
        "--screenshots-dir",
        type=str,
        default=SCREENSHOTS_DIR,
        help=f"Directory for SoM screenshots (default: {SCREENSHOTS_DIR})",
    )
    parser.add_argument(
        "--no-screenshots",
        action="store_true",
        help="Skip screenshot generation (faster, for testing)",
    )
    return parser.parse_args()


def save_som_screenshot(
    obs: dict[str, Any],
    screenshots_dir: Path,
    annotation_id: str,
    action_index: int,
) -> str | None:
    """Generate and save a SoM-annotated screenshot.

    Overlays BID labels on the browser screenshot using BrowserGym's
    overlay_som(). BID numbers on the screenshot match the axtree BIDs.

    Returns:
        Relative path to the saved screenshot, or None if generation fails.
    """
    screenshot = obs.get("screenshot")
    extra_props = obs.get("extra_element_properties")

    if screenshot is None or extra_props is None:
        return None

    try:
        som_screenshot = overlay_som(screenshot, extra_props)
        save_dir = screenshots_dir / annotation_id
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{action_index}.png"
        filepath = save_dir / filename
        Image.fromarray(som_screenshot).save(filepath)
        # Return path relative to repo root for ImageObservation
        return f"{SCREENSHOTS_DIR}/{annotation_id}/{filename}"
    except Exception as e:
        print(f"Warning: SoM screenshot failed for {annotation_id} "
              f"action {action_index}: {e}", file=sys.stderr)
        return None


def convert_to_trajectory(
    raw_data: dict[str, Any],
    axtree_gen: HTMLToAXTree,
    screenshots_dir: Path,
    generate_screenshots: bool,
) -> Trajectory:
    """Convert a raw Mind2Web trajectory to ADP standardized format.

    For each action step:
    1. Renders cleaned_html in BrowserGym → axtree + screenshot
    2. Resolves backend_node_id → BID via xpath evaluation
    3. Generates SoM-annotated screenshot
    4. Creates WebObservation (axtree + screenshot) and ApiAction (BID-based)

    Args:
        raw_data: Raw trajectory dict matching SchemaRaw.
        axtree_gen: Initialized HTMLToAXTree instance.
        screenshots_dir: Directory for SoM screenshots.
        generate_screenshots: Whether to generate SoM screenshots.

    Returns:
        Standardized Trajectory object.
    """
    data = SchemaRaw(**raw_data)
    content: list = []

    # Task instruction as first user observation
    content.append(TextObservation(content=data.confirmed_task, source="user"))

    # Initial goto action (construct URL from website name)
    website_url = f"https://www.{data.website}.com"
    content.append(ApiAction(function="goto", kwargs={"url": website_url}))

    # Track stats for this trajectory
    bid_resolved = 0
    bid_failed = 0
    actions_skipped = 0

    for action_index, action in enumerate(data.actions):
        step_id = f"{data.annotation_id}_{action_index}"

        # Render HTML and generate axtree
        axtree = axtree_gen.build_axtree(
            step_id, action.cleaned_html, data.annotation_id
        )

        if not axtree:
            print(f"Warning: axtree generation failed for {step_id}", file=sys.stderr)
            actions_skipped += 1
            continue

        # Generate SoM screenshot
        screenshot_path = None
        if generate_screenshots and axtree_gen.last_obs is not None:
            screenshot_path = save_som_screenshot(
                axtree_gen.last_obs,
                screenshots_dir,
                data.annotation_id,
                action_index,
            )

        # Build ImageObservation if screenshot was generated
        image_obs = None
        if screenshot_path:
            image_obs = ImageObservation(
                content=screenshot_path,
                source="environment",
                annotations=None,
            )

        # Create WebObservation with axtree and screenshot
        content.append(
            WebObservation(
                axtree=axtree,
                html=None,
                url=None,
                viewport_size=None,
                image_observation=image_obs,
            )
        )

        # Skip action if no target element (empty pos_candidates)
        if action.backend_node_id is None:
            actions_skipped += 1
            continue

        # Resolve backend_node_id → BID
        xpath = f"//*[@backend_node_id='{action.backend_node_id}']"
        bid = axtree_gen.get_bid(step_id, xpath, data.annotation_id)

        if bid is None:
            bid_failed += 1
            actions_skipped += 1
            continue

        bid_resolved += 1

        # Build action kwargs
        func_name = OPERATION_MAP.get(action.operation.op)
        if func_name is None:
            print(f"Warning: unknown operation '{action.operation.op}' "
                  f"for {step_id}", file=sys.stderr)
            actions_skipped += 1
            continue

        kwargs: dict[str, str] = {"bid": f'"{bid}"'}
        if func_name in ("type", "select") and action.operation.value:
            kwargs["value"] = action.operation.value

        content.append(ApiAction(function=func_name, kwargs=kwargs))

    # Log trajectory stats
    total = len(data.actions)
    print(
        f"  {data.annotation_id[:20]}... "
        f"({total} actions, {bid_resolved} resolved, "
        f"{bid_failed} BID failures, {actions_skipped} skipped)",
        file=sys.stderr,
    )

    details: dict[str, str] = {
        "website": data.website,
        "domain": data.domain,
        "subdomain": data.subdomain,
        "task_description": data.confirmed_task,
    }

    return Trajectory(
        id=data.annotation_id,
        content=content,
        details=details,
    )


def main() -> None:
    """Main conversion pipeline."""
    args = parse_args()
    filter_som = args.filter_som and not args.no_filter_som
    screenshots_dir = Path(args.screenshots_dir)
    generate_screenshots = not args.no_screenshots

    print(f"Initializing HTMLToAXTree (filter_som={filter_som})...", file=sys.stderr)
    axtree_gen = HTMLToAXTree(
        dataset="mind2web_multimodal",
        filter_som_only=filter_som,
        filter_visible_only=False,
    )

    record_count = 0
    error_count = 0

    for line in sys.stdin:
        try:
            raw_data = json.loads(line)
            trajectory = convert_to_trajectory(
                raw_data,
                axtree_gen,
                screenshots_dir,
                generate_screenshots,
            )
            print(trajectory.model_dump_json())
            record_count += 1
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            error_count += 1
            print(f"Warning: Skipping record: {e}", file=sys.stderr)

    axtree_gen.env.close()
    print(
        f"Processed {record_count} trajectories ({error_count} errors)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
