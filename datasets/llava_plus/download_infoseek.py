#!/usr/bin/env python3
"""Download InfoSeek/OVEN images needed by the llava_plus dataset.

Streams through the tar shards in the ychenNLP/oven HuggingFace dataset,
extracting only the images referenced by llava_plus. Handles mixed
extensions (.jpg vs .JPEG) by matching on filename stems.

Usage:
    python datasets/llava_plus/download_infoseek.py

    # Override paths via environment variables
    DATA_DIR=/custom/path python datasets/llava_plus/download_infoseek.py

Prerequisites:
    - infoseek_images_needed.txt generated from full_raw.jsonl
    - HuggingFace login: huggingface-cli login
    - Accepted OVEN terms at https://huggingface.co/datasets/ychenNLP/oven
"""

import logging
import os
import tarfile

from huggingface_hub import hf_hub_url, get_token
from tqdm import tqdm

from download_utils import (
    get_paths,
    get_stem,
    load_checkpoint,
    load_image_list,
    print_header,
    print_summary,
    save_checkpoint,
    stream_with_backoff,
)

logger = logging.getLogger(__name__)

DATASET_REPO = "ychenNLP/oven"
# OVEN image tar shards + the wikipedia images archive
SHARD_FILES = [
    *[f"shard{i:02d}.tar" for i in range(1, 9)],  # shard01.tar - shard08.tar
    "all_wikipedia_images.tar",                      # ~30 GB, contains remaining images
]


def process_shard(
    shard_file: str,
    needed_stems: set[str],
    stem_to_listname: dict[str, str],
    output_dir: str,
) -> tuple[int, int, list[tuple[str, str]]]:
    """Stream a single tar shard and extract matching images.

    Saves images using the filename from the image list (not the archive)
    to ensure raw_to_standardized.py path resolution works.

    Args:
        shard_file: Filename of the shard within the HF repo.
        needed_stems: Set of filename stems still needed.
        stem_to_listname: Maps stem to the filename from the image list.
        output_dir: Directory to save extracted images.

    Returns:
        Tuple of (saved_count, skipped_count, errors list).
    """
    url = hf_hub_url(DATASET_REPO, shard_file, repo_type="dataset")
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    response = stream_with_backoff(url, headers=headers)

    saved = 0
    skipped = 0
    errors: list[tuple[str, str]] = []

    with tarfile.open(fileobj=response.raw, mode="r|") as tar:
        for member in tar:
            if not member.isfile():
                continue

            basename = os.path.basename(member.name)
            stem = get_stem(basename)

            if stem not in needed_stems:
                continue

            # Save using the filename from the image list
            target_name = stem_to_listname.get(stem, basename)
            output_path = os.path.join(output_dir, target_name)

            if os.path.exists(output_path):
                skipped += 1
                continue

            f = tar.extractfile(member)
            if not f:
                errors.append((target_name, "extractfile returned None"))
                continue

            try:
                with open(output_path, "wb") as out:
                    out.write(f.read())
                saved += 1
            except Exception as e:
                if os.path.exists(output_path):
                    os.remove(output_path)
                errors.append((target_name, f"{type(e).__name__}: {e}"))

    return saved, skipped, errors


def main() -> None:
    """Download InfoSeek/OVEN images referenced by llava_plus."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    llava_plus_dir, image_list_file, output_dir, failed_list_path = get_paths("infoseek")
    checkpoint_file = os.path.join(llava_plus_dir, "infoseek_last_shard.txt")

    print_header(
        "InfoSeek/OVEN Image Downloader for llava_plus",
        data_directory=os.environ.get("DATA_DIR", "default"),
        image_list=image_list_file,
        output_directory=output_dir,
        dataset=DATASET_REPO,
        shards=f"{len(SHARD_FILES)} ({SHARD_FILES[0]} - {SHARD_FILES[-1]})",
    )

    # Match by stem since image list may say .jpg but archive has .JPEG
    all_files, already, remaining = load_image_list(
        image_list_file, output_dir, match_by_stem=True
    )

    # Build stem → image-list-filename lookup
    stem_to_listname = {get_stem(fn): fn for fn in all_files}

    print(f"Total images needed: {len(set(all_files)):,}")
    print(f"Already downloaded: {len(already):,}")
    print(f"Remaining: {len(remaining):,}")
    print()

    if not remaining:
        print("All images already downloaded!")
        return

    # Check HF authentication
    token = get_token()
    if not token:
        print("ERROR: No HuggingFace token found.")
        print("The OVEN dataset is gated and requires authentication.")
        print("Run: huggingface-cli login")
        print(f"And accept terms at: https://huggingface.co/datasets/{DATASET_REPO}")
        return

    # Checkpoint: resume from last scanned shard
    start_shard = load_checkpoint(checkpoint_file)
    if start_shard > 0 and remaining:
        print(f"Resuming from shard index {start_shard} (checkpoint found)")
        print()

    total_saved = 0
    total_skipped = len(already)
    total_failed = 0
    all_failed: list[tuple[str, str]] = []
    found_stems: set[str] = set()
    remaining_stems = set(remaining)

    print(f"Scanning {len(SHARD_FILES)} shards for {len(remaining_stems):,} images...")
    print()

    for shard_idx in tqdm(
        range(start_shard, len(SHARD_FILES)),
        initial=start_shard, total=len(SHARD_FILES),
        desc="Shards", unit="shard",
    ):
        if not remaining_stems - found_stems:
            print(f"\nAll needed images found after shard {shard_idx}!")
            break

        shard_file = SHARD_FILES[shard_idx]

        try:
            saved, skipped, errors = process_shard(
                shard_file,
                remaining_stems - found_stems,
                stem_to_listname,
                output_dir,
            )

            total_saved += saved
            total_skipped += skipped
            total_failed += len(errors)
            all_failed.extend(errors)

            # Track found stems
            newly_found = {
                get_stem(fn)
                for fn in os.listdir(output_dir)
                if get_stem(fn) in (remaining_stems - found_stems)
            }
            found_stems |= newly_found

            logger.info(
                "Shard %s: saved %d, skipped %d, errors %d (remaining: %d)",
                shard_file, saved, skipped, len(errors),
                len(remaining_stems) - len(found_stems),
            )

            save_checkpoint(checkpoint_file, shard_idx)

        except Exception as e:
            logger.error("Failed to process %s: %s: %s",
                         shard_file, type(e).__name__, e)

    not_found = remaining_stems - found_stems
    if not_found:
        print(f"\nNot found in any shard: {len(not_found):,}")
        print("Images not found (first 20):")
        for stem in list(not_found)[:20]:
            print(f"  {stem_to_listname.get(stem, stem)}")
        for stem in not_found:
            all_failed.append((stem_to_listname.get(stem, stem), "not found in any shard"))
        total_failed += len(not_found)

    print_summary(total_saved, total_skipped, total_failed, all_failed, failed_list_path)

    # Clean up checkpoint on successful completion
    if not not_found and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)


if __name__ == "__main__":
    main()
