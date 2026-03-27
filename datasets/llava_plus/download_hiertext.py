#!/usr/bin/env python3
"""Download HierText images needed by the llava_plus dataset.

HierText images come from the Open Images OCR subset, stored as 3 tar archives
on AWS S3. This script streams through each archive and extracts only the
images referenced by llava_plus.

Usage:
    python datasets/llava_plus/download_hiertext.py

    # Override paths via environment variables
    DATA_DIR=/custom/path python datasets/llava_plus/download_hiertext.py
"""

import logging
import os
import tarfile

from tqdm import tqdm

from download_utils import (
    get_paths,
    load_image_list,
    print_header,
    print_summary,
    stream_with_backoff,
)

logger = logging.getLogger(__name__)

# Open Images OCR archives (train ~2.6 GB, validation ~550 MB, test ~512 MB)
HIERTEXT_ARCHIVES = [
    ("train", "https://open-images-dataset.s3.amazonaws.com/ocr/train.tgz"),
    ("validation", "https://open-images-dataset.s3.amazonaws.com/ocr/validation.tgz"),
    ("test", "https://open-images-dataset.s3.amazonaws.com/ocr/test.tgz"),
]


def extract_from_archive(
    url: str,
    archive_name: str,
    needed: set[str],
    output_dir: str,
) -> tuple[int, int, list[tuple[str, str]]]:
    """Stream a .tgz archive and extract matching images.

    Args:
        url: URL of the archive.
        archive_name: Human-readable name for logging (e.g., "train").
        needed: Set of filenames still needed.
        output_dir: Directory to save extracted images.

    Returns:
        Tuple of (saved_count, skipped_count, errors list [(filename, msg)]).
    """
    response = stream_with_backoff(url)

    saved = 0
    skipped = 0
    errors: list[tuple[str, str]] = []

    with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue

            basename = os.path.basename(member.name)
            if basename not in needed:
                continue

            output_path = os.path.join(output_dir, basename)
            if os.path.exists(output_path):
                skipped += 1
                continue

            f = tar.extractfile(member)
            if not f:
                errors.append((basename, "extractfile returned None"))
                continue

            try:
                with open(output_path, "wb") as out:
                    out.write(f.read())
                saved += 1
            except Exception as e:
                if os.path.exists(output_path):
                    os.remove(output_path)
                errors.append((basename, f"{type(e).__name__}: {e}"))

    return saved, skipped, errors


def main() -> None:
    """Download HierText images referenced by llava_plus."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    llava_plus_dir, image_list_file, output_dir, failed_list_path = get_paths("hiertext")

    print_header(
        "HierText Image Downloader for llava_plus",
        data_directory=os.environ.get("DATA_DIR", "default"),
        image_list=image_list_file,
        output_directory=output_dir,
        archives=f"{len(HIERTEXT_ARCHIVES)} (train, validation, test)",
    )

    all_files, already, remaining = load_image_list(image_list_file, output_dir)

    print(f"Total images needed: {len(set(all_files)):,}")
    print(f"Already downloaded: {len(already):,}")
    print(f"Remaining: {len(remaining):,}")
    print()

    if not remaining:
        print("All images already downloaded!")
        return

    total_saved = 0
    total_skipped = len(already)
    total_failed = 0
    all_failed: list[tuple[str, str]] = []
    still_needed = set(remaining)

    for archive_name, url in tqdm(HIERTEXT_ARCHIVES, desc="Archives", unit="archive"):
        if not still_needed:
            print(f"All needed images found, skipping {archive_name}")
            continue

        logger.info("Processing %s archive (%d images still needed)...",
                     archive_name, len(still_needed))

        try:
            saved, skipped, errors = extract_from_archive(
                url, archive_name, still_needed, output_dir
            )

            total_saved += saved
            total_skipped += skipped
            total_failed += len(errors)
            all_failed.extend(errors)

            # Remove found images from needed set
            found_in_archive = set(os.listdir(output_dir)) & still_needed
            still_needed -= found_in_archive

            logger.info(
                "%s: saved %d, skipped %d, errors %d (remaining: %d)",
                archive_name, saved, skipped, len(errors), len(still_needed),
            )

        except Exception as e:
            logger.error("Failed to process %s archive: %s: %s",
                         archive_name, type(e).__name__, e)

    if still_needed:
        print(f"\nNot found in any archive: {len(still_needed):,}")
        print("Images not found (first 20):")
        for fn in list(still_needed)[:20]:
            print(f"  {fn}")
        # Count not-found as failures
        for fn in still_needed:
            all_failed.append((fn, "not found in any archive"))
        total_failed += len(still_needed)

    print_summary(total_saved, total_skipped, total_failed, all_failed, failed_list_path)


if __name__ == "__main__":
    main()
