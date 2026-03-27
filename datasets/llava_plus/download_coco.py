#!/usr/bin/env python3
"""Download COCO images needed by the llava_plus dataset.

Downloads the specific COCO images referenced in llava_plus rather than
the entire COCO dataset. Tries train2017 first, falls back to val2017.

Usage:
    python datasets/llava_plus/download_coco.py

    # Override paths via environment variables
    DATA_DIR=/custom/path python datasets/llava_plus/download_coco.py
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from download_utils import (
    download_with_backoff_multi_url,
    get_paths,
    load_image_list,
    print_header,
    print_summary,
)

# COCO image URLs — try train first, then val
COCO_URLS = [
    "http://images.cocodataset.org/train2017/{}",
    "http://images.cocodataset.org/val2017/{}",
]

MAX_WORKERS = 8


def download_image(filename: str, output_dir: str) -> tuple[str, bool, str]:
    """Download a single COCO image.

    Args:
        filename: Image filename (e.g., "000000000009.jpg").
        output_dir: Directory to save the image.

    Returns:
        Tuple of (filename, success, message).
    """
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        return (filename, True, "already exists")

    urls = [pattern.format(filename) for pattern in COCO_URLS]
    success, msg = download_with_backoff_multi_url(urls, output_path)
    return (filename, success, msg)


def main() -> None:
    """Download COCO images referenced by llava_plus."""
    llava_plus_dir, image_list_file, output_dir, failed_list_path = get_paths("coco")

    print_header(
        "COCO Image Downloader for llava_plus",
        data_directory=os.environ.get("DATA_DIR", "default"),
        image_list=image_list_file,
        output_directory=output_dir,
        parallel_workers=str(MAX_WORKERS),
    )

    all_files, already, remaining = load_image_list(image_list_file, output_dir)

    print(f"Total images: {len(all_files):,}")
    print(f"Already downloaded: {len(already):,}")
    print(f"Remaining: {len(remaining):,}")
    print()

    if not remaining:
        print("All images already downloaded!")
        return

    # Download only remaining files in parallel
    success_count = 0
    skip_count = len(already)
    fail_count = 0
    failed_images: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_image, img, output_dir): img
            for img in remaining
        }

        with tqdm(total=len(remaining), desc="Downloading", unit="img") as pbar:
            for future in as_completed(futures):
                filename, success, msg = future.result()

                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    failed_images.append((filename, msg))

                pbar.update(1)
                pbar.set_postfix(
                    downloaded=success_count, failed=fail_count
                )

    print_summary(success_count, skip_count, fail_count, failed_images, failed_list_path)


if __name__ == "__main__":
    main()
