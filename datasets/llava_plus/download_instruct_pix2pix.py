#!/usr/bin/env python3
"""Download InstructPix2Pix images needed by the llava_plus dataset.

The InstructPix2Pix clip-filtered dataset is distributed as 30 zip shards on
the Berkeley server. This script downloads each shard to a temp directory,
extracts only the needed images, then deletes the shard.

Usage:
    python datasets/llava_plus/download_instruct_pix2pix.py

    # Override paths via environment variables
    DATA_DIR=/custom/path python datasets/llava_plus/download_instruct_pix2pix.py
"""

import logging
import os
import shutil
import zipfile

from tqdm import tqdm

from download_utils import (
    download_with_backoff,
    get_paths,
    load_checkpoint,
    print_header,
    print_summary,
    save_checkpoint,
)

logger = logging.getLogger(__name__)

# Berkeley server hosts 30 zip shards (shard-00.zip - shard-29.zip)
BASE_URL = "http://instruct-pix2pix.eecs.berkeley.edu/clip-filtered-dataset"
NUM_SHARDS = 30


def extract_from_zip(
    zip_path: str,
    needed_paths: set[str],
    output_dir: str,
) -> tuple[int, int, list[tuple[str, str]]]:
    """Extract matching images from a zip file.

    Args:
        zip_path: Path to the local zip file.
        needed_paths: Set of relative image paths to look for
            (e.g., "0183936/1314947222_0.jpg").
        output_dir: Base directory to save images (preserving subdirectories).

    Returns:
        Tuple of (saved_count, skipped_count, errors list).
    """
    saved = 0
    skipped = 0
    errors: list[tuple[str, str]] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for entry in zf.namelist():
            if entry.endswith("/"):
                continue  # Skip directory entries

            if entry not in needed_paths:
                continue

            output_path = os.path.join(output_dir, entry)
            if os.path.exists(output_path):
                skipped += 1
                continue

            try:
                # Create parent directory for subdirectory structure
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                with zf.open(entry) as src, open(output_path, "wb") as dst:
                    dst.write(src.read())
                saved += 1
            except Exception as e:
                if os.path.exists(output_path):
                    os.remove(output_path)
                errors.append((entry, f"{type(e).__name__}: {e}"))

    return saved, skipped, errors


def main() -> None:
    """Download InstructPix2Pix images referenced by llava_plus."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    llava_plus_dir, image_list_file, output_dir, failed_list_path = get_paths(
        "instruct-pix2pix"
    )
    checkpoint_file = os.path.join(llava_plus_dir, "instruct_pix2pix_last_shard.txt")
    tmp_dir = os.path.join(llava_plus_dir, "tmp")

    print_header(
        "InstructPix2Pix Image Downloader for llava_plus",
        data_directory=os.environ.get("DATA_DIR", "default"),
        image_list=image_list_file,
        output_directory=output_dir,
        temp_directory=tmp_dir,
        shards=f"{NUM_SHARDS} ({BASE_URL})",
    )

    # Can't use load_image_list here because image paths contain subdirectories
    # (e.g., "0183936/1314947222_0.jpg") and os.listdir only sees top-level entries
    if not os.path.exists(image_list_file):
        print(f"ERROR: Image list not found: {image_list_file}")
        print("Generate it from full_raw.jsonl with jq. See README for commands.")
        return

    with open(image_list_file) as f:
        all_files = [line.strip() for line in f if line.strip()]

    needed_paths = set(all_files)
    os.makedirs(output_dir, exist_ok=True)

    # Check each path individually since files are in subdirectories
    already = {p for p in needed_paths if os.path.exists(os.path.join(output_dir, p))}
    remaining = needed_paths - already

    print(f"Total images needed: {len(needed_paths):,}")
    print(f"Already downloaded: {len(already):,}")
    print(f"Remaining: {len(remaining):,}")
    print()

    if not remaining:
        print("All images already downloaded!")
        return

    # Check available disk space for temp downloads (~14 GB per shard)
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        free_space = shutil.disk_usage(tmp_dir).free
        min_space = 16 * 1024 * 1024 * 1024  # 16 GB
        if free_space < min_space:
            print(f"WARNING: Only {free_space / (1024**3):.1f} GB free in {tmp_dir}")
            print("Each shard is ~14 GB. Downloads may fail if space runs out.")
            print()
    except OSError:
        pass  # disk_usage may fail on some filesystems

    # Checkpoint: resume from last processed shard
    start_shard = load_checkpoint(checkpoint_file)
    if start_shard > 0 and remaining:
        print(f"Resuming from shard {start_shard:02d} (checkpoint found)")
        print()

    total_saved = 0
    total_skipped = len(already)
    total_failed = 0
    all_failed: list[tuple[str, str]] = []
    found_paths: set[str] = set()

    print(f"Processing {NUM_SHARDS} shards (~14 GB each, downloaded then deleted)...")
    print()

    for shard_idx in tqdm(
        range(start_shard, NUM_SHARDS),
        initial=start_shard, total=NUM_SHARDS,
        desc="Shards", unit="shard",
    ):
        still_needed = remaining - found_paths
        if not still_needed:
            print(f"\nAll needed images found after shard {shard_idx:02d}!")
            break

        shard_url = f"{BASE_URL}/shard-{shard_idx:02d}.zip"
        shard_local = os.path.join(tmp_dir, f"shard-{shard_idx:02d}.zip")

        try:
            # Download the full zip shard
            logger.info("Downloading shard %02d...", shard_idx)
            success, msg = download_with_backoff(
                shard_url, shard_local, max_retries=5, base_delay=5.0, max_delay=120.0
            )

            if not success:
                logger.error("Failed to download shard %02d: %s", shard_idx, msg)
                # Don't save checkpoint on download failure — allows retry on next run
                continue

            # Extract matching images
            logger.info("Extracting from shard %02d...", shard_idx)
            saved, skipped, errors = extract_from_zip(
                shard_local, still_needed, output_dir
            )

            total_saved += saved
            total_skipped += skipped
            total_failed += len(errors)
            all_failed.extend(errors)

            # Track what was found by checking output dir
            for path in still_needed:
                if os.path.exists(os.path.join(output_dir, path)):
                    found_paths.add(path)

            logger.info(
                "Shard %02d: saved %d, skipped %d, errors %d (remaining: %d)",
                shard_idx, saved, skipped, len(errors),
                len(remaining) - len(found_paths),
            )

            save_checkpoint(checkpoint_file, shard_idx)

        except Exception as e:
            logger.error("Failed to process shard %02d: %s: %s",
                         shard_idx, type(e).__name__, e)
        finally:
            # Always clean up the temp zip file
            if os.path.exists(shard_local):
                os.remove(shard_local)
                logger.info("Cleaned up temp file: %s", shard_local)

    not_found = remaining - found_paths
    if not_found:
        print(f"\nNot found in any shard: {len(not_found):,}")
        print("Images not found (first 20):")
        for path in list(not_found)[:20]:
            print(f"  {path}")
        for path in not_found:
            all_failed.append((path, "not found in any shard"))
        total_failed += len(not_found)

    print_summary(total_saved, total_skipped, total_failed, all_failed, failed_list_path)

    # Clean up checkpoint and tmp dir on successful completion
    if not not_found and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    # Remove tmp dir if empty
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass  # Not empty or doesn't exist


if __name__ == "__main__":
    main()
