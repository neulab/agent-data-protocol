#!/usr/bin/env python3
"""Download Visual Genome images needed by the llava_plus dataset.

Downloads the specific VG images referenced in llava_plus. Tries VG_100K first,
falls back to VG_100K_2.

Usage:
    python datasets/llava_plus/download_vg.py

    # Override paths via environment variables
    DATA_DIR=/custom/path python datasets/llava_plus/download_vg.py
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

# Visual Genome URLs — try VG_100K first, then VG_100K_2
VG_URLS = [
    "https://cs.stanford.edu/people/rak248/VG_100K/{}",
    "https://cs.stanford.edu/people/rak248/VG_100K_2/{}",
]

MAX_WORKERS = 8


def download_image(filename: str, output_dir: str) -> tuple[str, bool, str]:
    """Download a single Visual Genome image.

    Args:
        filename: Image filename (e.g., "2373367.jpg").
        output_dir: Directory to save the image.

    Returns:
        Tuple of (filename, success, message).
    """
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        return (filename, True, "already exists")

    urls = [pattern.format(filename) for pattern in VG_URLS]
    success, msg = download_with_backoff_multi_url(urls, output_path)
    return (filename, success, msg)


def main() -> None:
    """Download Visual Genome images referenced by llava_plus."""
    llava_plus_dir, image_list_file, output_dir, failed_list_path = get_paths("vg")

    print_header(
        "Visual Genome Image Downloader for llava_plus",
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
