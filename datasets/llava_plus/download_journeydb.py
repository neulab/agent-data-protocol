#!/usr/bin/env python3
"""Download JourneyDB images needed by the llava_plus dataset.

Streams through the official JourneyDB/JourneyDB .tgz archives on HuggingFace,
extracting only the needed images. Uses the HF cache if archives are already
downloaded (e.g., shared cluster cache).

Usage:
    python datasets/llava_plus/download_journeydb.py

    # Override paths via environment variables
    DATA_DIR=/custom/path python datasets/llava_plus/download_journeydb.py

Prerequisites:
    - journeydb_images_needed.txt generated from full_raw.jsonl
    - HuggingFace login: huggingface-cli login
    - Accepted JourneyDB terms at https://huggingface.co/datasets/JourneyDB/JourneyDB
"""

import io
import logging
import os
import tarfile
from collections import Counter

from huggingface_hub import hf_hub_download, hf_hub_url, get_token
from PIL import Image as PILImage
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

DATASET_REPO = "JourneyDB/JourneyDB"
NUM_TRAIN_SHARDS = 200  # 000.tgz through 199.tgz


def save_from_tar(
    tar: tarfile.TarFile, member: tarfile.TarInfo,
    uuid: str, output_dir: str,
) -> tuple[bool, str]:
    """Extract a single image from a tar archive and save as PNG.

    Args:
        tar: Open tar archive.
        member: Tar member to extract.
        uuid: Image UUID (used for output filename).
        output_dir: Directory to save the image.

    Returns:
        Tuple of (success, message).
    """
    output_path = os.path.join(output_dir, f"{uuid}.png")
    if os.path.exists(output_path):
        return True, "already exists"

    f = tar.extractfile(member)
    if not f:
        return False, "extractfile returned None"

    try:
        img_data = f.read()
        image = PILImage.open(io.BytesIO(img_data))
        image.load()  # Force full pixel decode to catch truncation
        image.save(output_path, "PNG")
        return True, "saved"
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        return False, f"{type(e).__name__}: {e}"


def extract_from_local(
    local_path: str, needed_uuids: set[str], output_dir: str,
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Extract matching images from a locally cached .tgz file.

    Args:
        local_path: Path to local .tgz archive.
        needed_uuids: Set of UUIDs to look for.
        output_dir: Directory to save extracted images.

    Returns:
        Tuple of (found dict {uuid: archive_name}, errors list [(uuid, msg)]).
    """
    found: dict[str, str] = {}
    errors: list[tuple[str, str]] = []
    with tarfile.open(local_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = os.path.basename(member.name)
            uuid = get_stem(name)
            if uuid in needed_uuids:
                success, msg = save_from_tar(tar, member, uuid, output_dir)
                if success:
                    found[uuid] = name
                else:
                    errors.append((uuid, msg))
    return found, errors


def extract_from_stream(
    archive_path: str, needed_uuids: set[str], output_dir: str,
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Stream a .tgz archive over HTTP and extract matching images.

    Args:
        archive_path: HF repo-relative path to the archive.
        needed_uuids: Set of UUIDs to look for.
        output_dir: Directory to save extracted images.

    Returns:
        Tuple of (found dict {uuid: archive_name}, errors list [(uuid, msg)]).
    """
    url = hf_hub_url(DATASET_REPO, archive_path, repo_type="dataset")
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    response = stream_with_backoff(url, headers=headers)

    found: dict[str, str] = {}
    errors: list[tuple[str, str]] = []
    with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = os.path.basename(member.name)
            uuid = get_stem(name)
            if uuid in needed_uuids:
                success, msg = save_from_tar(tar, member, uuid, output_dir)
                if success:
                    found[uuid] = name
                else:
                    errors.append((uuid, msg))
    return found, errors


def process_shard(
    shard_idx: int, needed_uuids: set[str], output_dir: str,
) -> tuple[dict[str, str], list[tuple[str, str]], str]:
    """Process a single .tgz shard, using cache if available, else streaming.

    Args:
        shard_idx: Shard index (0-199).
        needed_uuids: Set of UUIDs still needed.
        output_dir: Directory to save extracted images.

    Returns:
        Tuple of (found dict, errors list, source "cache"|"stream").
    """
    archive_path = f"data/train/imgs/{shard_idx:03d}.tgz"

    # Try reading from HF cache first (fast, no download)
    try:
        local_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=archive_path,
            repo_type="dataset",
            local_files_only=True,
        )
        logger.info("Shard %03d: reading from cache", shard_idx)
        found, errors = extract_from_local(local_path, needed_uuids, output_dir)
        return found, errors, "cache"
    except Exception:
        pass  # Not in cache

    # Stream over HTTP without caching
    logger.info("Shard %03d: streaming over HTTP", shard_idx)
    found, errors = extract_from_stream(archive_path, needed_uuids, output_dir)
    return found, errors, "stream"


def main() -> None:
    """Download JourneyDB images referenced by llava_plus."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    llava_plus_dir, image_list_file, output_dir, failed_list_path = get_paths("journeydb")
    checkpoint_file = os.path.join(llava_plus_dir, "journeydb_last_shard.txt")

    print_header(
        "JourneyDB Image Downloader for llava_plus",
        data_directory=os.environ.get("DATA_DIR", "default"),
        image_list=image_list_file,
        output_directory=output_dir,
        dataset=DATASET_REPO,
        shards=f"{NUM_TRAIN_SHARDS} (000.tgz - {NUM_TRAIN_SHARDS - 1:03d}.tgz)",
    )

    # Use match_by_stem since JourneyDB images are saved as .png but list may have other extensions
    all_files, already, remaining = load_image_list(
        image_list_file, output_dir, match_by_stem=True
    )
    needed_uuids = {get_stem(fn) for fn in all_files}

    print(f"Total images needed: {len(needed_uuids):,}")
    print(f"Already downloaded: {len(already):,}")
    print(f"Remaining: {len(remaining):,}")
    print()

    if not remaining:
        print("All images already downloaded!")
        return

    # Check HF authentication
    token = get_token()
    if not token:
        print("WARNING: No HuggingFace token found.")
        print("The JourneyDB dataset is gated and requires authentication.")
        print("Run: huggingface-cli login")
        print(f"And accept terms at: https://huggingface.co/datasets/{DATASET_REPO}")
        print()

    # Checkpoint: resume from last scanned shard
    start_shard = load_checkpoint(checkpoint_file)
    if start_shard > 0 and remaining:
        print(f"Resuming from shard {start_shard:03d} (checkpoint found)")
        print()

    # Track progress
    success_count = 0
    skip_count = len(already)
    fail_count = 0
    failed_images: list[tuple[str, str]] = []
    found_uuids: set[str] = set()
    shard_errors: list[tuple[int, str]] = []
    cache_hits = 0
    stream_count = 0

    remaining_uuids = set(remaining)

    print(f"Scanning {NUM_TRAIN_SHARDS} shards for {len(remaining_uuids):,} images...")
    print()

    for shard_idx in tqdm(
        range(start_shard, NUM_TRAIN_SHARDS),
        initial=start_shard, total=NUM_TRAIN_SHARDS,
        desc="Shards", unit="shard", mininterval=10,
    ):
        if not remaining_uuids - found_uuids:
            print(f"\nAll needed images found after shard {shard_idx:03d}!")
            break

        try:
            found, errors, source = process_shard(
                shard_idx, remaining_uuids - found_uuids, output_dir
            )

            if source == "cache":
                cache_hits += 1
            else:
                stream_count += 1

            for uuid in found:
                success_count += 1
                found_uuids.add(uuid)

            for uuid, msg in errors:
                fail_count += 1
                failed_images.append((f"{uuid}.png", msg))
                logger.warning(
                    "Failed to extract %s from shard %03d: %s", uuid, shard_idx, msg
                )

            if found:
                logger.info(
                    "Shard %03d: found %d images (total: %d/%d, remaining: %d)",
                    shard_idx, len(found), len(found_uuids),
                    len(remaining_uuids), len(remaining_uuids) - len(found_uuids),
                )

            save_checkpoint(checkpoint_file, shard_idx)

        except Exception as e:
            shard_errors.append((shard_idx, f"{type(e).__name__}: {e}"))
            logger.error(
                "Failed to process shard %03d: %s: %s",
                shard_idx, type(e).__name__, e,
            )

    # Include not-found in failure counts
    not_found = remaining_uuids - found_uuids
    if not_found:
        for uuid in not_found:
            failed_images.append((f"{uuid}.png", "not found in any shard"))
        fail_count += len(not_found)

    # Summary
    print_summary(success_count, skip_count, fail_count, failed_images, failed_list_path)

    print(f"Shards from cache: {cache_hits:,}")
    print(f"Shards streamed: {stream_count:,}")
    print(f"Shard errors: {len(shard_errors):,}")

    if shard_errors:
        print()
        error_types = Counter(e.split(":")[0] for _, e in shard_errors)
        print("Shard errors by type:")
        for etype, count in error_types.most_common():
            print(f"  {etype}: {count:,}")
        print("Shard errors (first 10):")
        for idx, msg in shard_errors[:10]:
            print(f"  shard {idx:03d}: {msg}")

    if not_found:
        print(f"\nNot found in dataset: {len(not_found):,}")
        print("Images not found (first 20):")
        for uuid in list(not_found)[:20]:
            print(f"  {uuid}.png")

    # Clean up checkpoint on successful completion
    if not not_found and not shard_errors and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)


if __name__ == "__main__":
    main()
