"""Shared utilities for llava_plus image download scripts.

Provides common path configuration, image list loading, progress reporting,
and HTTP download/streaming with exponential backoff.
"""

import os
import random
import sys
import time
import urllib.error
import urllib.request

import requests
from tqdm import tqdm

def get_paths(source_name: str) -> tuple[str, str, str, str]:
    """Return standard paths for a given image source.

    Args:
        source_name: Image source identifier (e.g., "coco", "vg").

    Returns:
        Tuple of (llava_plus_dir, image_list_file, output_dir, failed_list_path).

    Raises:
        RuntimeError: If DATA_DIR environment variable is not set.
    """
    data_dir = os.environ.get("DATA_DIR")
    if not data_dir:
        raise RuntimeError("DATA_DIR environment variable must be set")
    llava_plus_dir = os.path.join(data_dir, "llava_plus")
    image_list_file = os.path.join(llava_plus_dir, f"{source_name}_images_needed.txt")
    output_dir = os.path.join(llava_plus_dir, "images", source_name)
    failed_list_path = os.path.join(llava_plus_dir, f"{source_name}_download_failed.txt")
    return llava_plus_dir, image_list_file, output_dir, failed_list_path


def get_stem(filename: str) -> str:
    """Extract filename stem by stripping the extension.

    Args:
        filename: A filename like "image.jpg" or "uuid.png".

    Returns:
        The stem, e.g. "image" or "uuid".
    """
    if "." in filename:
        return filename.rsplit(".", 1)[0]
    return filename


def load_image_list(
    image_list_file: str,
    output_dir: str,
    match_by_stem: bool = False,
) -> tuple[list[str], set[str], set[str]]:
    """Load the needed-images list and compute what's already downloaded.

    Args:
        image_list_file: Path to text file with one filename per line.
        output_dir: Directory where downloaded images are saved.
        match_by_stem: If True, match existing files by stem (ignoring extension).
            Useful when archive extensions differ from the image list.

    Returns:
        Tuple of (all_filenames, already_downloaded_set, remaining_set).
        Sets contain filenames (or stems if match_by_stem).
    """
    if not os.path.exists(image_list_file):
        print(f"ERROR: Image list not found: {image_list_file}")
        print("Generate it from full_raw.jsonl with jq. See README for commands.")
        sys.exit(1)

    with open(image_list_file) as f:
        all_files = [line.strip() for line in f if line.strip()]

    os.makedirs(output_dir, exist_ok=True)

    if match_by_stem:
        needed = {get_stem(fn) for fn in all_files}
        existing_files = set(os.listdir(output_dir)) if os.path.isdir(output_dir) else set()
        existing = {get_stem(f) for f in existing_files}
        already = needed & existing
        remaining = needed - already
    else:
        needed = set(all_files)
        existing = set(os.listdir(output_dir)) if os.path.isdir(output_dir) else set()
        already = needed & existing
        remaining = needed - already

    return all_files, already, remaining


def print_header(title: str, **info: str) -> None:
    """Print a standardized download header block.

    Args:
        title: Header title (e.g., "COCO Image Downloader").
        **info: Key-value pairs to display (e.g., data_dir="/data/...").
    """
    width = 60
    print("=" * width)
    print(title)
    print("=" * width)
    for key, value in info.items():
        # Convert snake_case keys to readable labels
        label = key.replace("_", " ").capitalize()
        print(f"{label}: {value}")
    print()


def print_summary(
    success: int,
    skipped: int,
    failed: int,
    failed_images: list[tuple[str, str]],
    failed_list_path: str,
) -> None:
    """Print a standardized completion summary and save failed list.

    Args:
        success: Number of newly downloaded images.
        skipped: Number of already-existing images skipped.
        failed: Number of failed downloads.
        failed_images: List of (filename, error_message) tuples.
        failed_list_path: Path to write the failed images file.
    """
    print()
    print("=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print(f"Downloaded: {success:,}")
    print(f"Skipped (already exist): {skipped:,}")
    print(f"Failed: {failed:,}")

    if failed_images:
        print()
        print("Failed images (first 20):")
        for filename, error in failed_images[:20]:
            print(f"  {filename}: {error}")

        with open(failed_list_path, "w") as f:
            for filename, error in failed_images:
                f.write(f"{filename}\t{error}\n")
        print(f"\nFull failed list saved to: {failed_list_path}")


def _backoff_delay(attempt: int, base_delay: float, max_delay: float) -> float:
    """Compute delay with exponential backoff and jitter.

    Args:
        attempt: Zero-based attempt index.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay cap in seconds.

    Returns:
        Delay in seconds (with random jitter).
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    # Add 0-25% jitter to avoid thundering herd
    jitter = delay * random.uniform(0, 0.25)
    return delay + jitter


def download_with_backoff(
    url: str,
    output_path: str,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> tuple[bool, str]:
    """Download a URL to a file with exponential backoff.

    Retries on transient errors with exponentially increasing delays
    (1s, 2s, 4s, 8s, 16s, capped at max_delay) plus random jitter.
    Returns immediately on 404 without retrying.

    Args:
        url: URL to download.
        output_path: Local file path to save to.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial retry delay in seconds.
        max_delay: Maximum retry delay in seconds.

    Returns:
        Tuple of (success, message).
    """
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, output_path)
            return True, "downloaded"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Clean up partial file and return immediately
                if os.path.exists(output_path):
                    os.remove(output_path)
                return False, "404"
            if attempt < max_retries - 1:
                # Clean up partial file before retry
                if os.path.exists(output_path):
                    os.remove(output_path)
                time.sleep(_backoff_delay(attempt, base_delay, max_delay))
            else:
                if os.path.exists(output_path):
                    os.remove(output_path)
                return False, f"HTTP {e.code}"
        except Exception as e:
            if attempt < max_retries - 1:
                if os.path.exists(output_path):
                    os.remove(output_path)
                time.sleep(_backoff_delay(attempt, base_delay, max_delay))
            else:
                if os.path.exists(output_path):
                    os.remove(output_path)
                return False, str(e)

    return False, "max retries exceeded"


def download_with_backoff_multi_url(
    urls: list[str],
    output_path: str,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> tuple[bool, str]:
    """Try multiple URLs in order, with backoff on each.

    For each URL, retries on transient errors. Moves to the next URL on 404.
    Used by COCO (train/val) and VG (VG_100K/VG_100K_2) downloaders.

    Args:
        urls: List of URLs to try in order.
        output_path: Local file path to save to.
        max_retries: Maximum retry attempts per URL.
        base_delay: Initial retry delay in seconds.
        max_delay: Maximum retry delay in seconds.

    Returns:
        Tuple of (success, message).
    """
    for url in urls:
        success, msg = download_with_backoff(
            url, output_path, max_retries, base_delay, max_delay
        )
        if success:
            return True, msg
        if msg != "404":
            # Non-404 failure (e.g., timeout, server error) — already retried
            return False, msg
        # 404 — try next URL

    # Clean up any partial file
    if os.path.exists(output_path):
        os.remove(output_path)
    return False, f"not found at any of {len(urls)} URLs"


def stream_with_backoff(
    url: str,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    headers: dict[str, str] | None = None,
    timeout: int = 120,
) -> requests.Response:
    """Open a streaming HTTP connection with exponential backoff.

    For archive-streaming scripts that need to read tar/zip archives
    from a URL without downloading the entire file first.

    Args:
        url: URL to stream from.
        max_retries: Maximum retry attempts.
        base_delay: Initial retry delay in seconds.
        max_delay: Maximum retry delay in seconds.
        headers: Optional HTTP headers (e.g., for auth).
        timeout: Connection timeout in seconds.

    Returns:
        A streaming requests.Response object.

    Raises:
        requests.HTTPError: If all retries are exhausted.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url, headers=headers or {}, stream=True, timeout=timeout
            )
            response.raise_for_status()
            return response
        except (requests.RequestException, IOError) as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(_backoff_delay(attempt, base_delay, max_delay))

    raise last_error  # type: ignore[misc]


def load_checkpoint(checkpoint_file: str) -> int:
    """Load the last fully processed shard index from a checkpoint file.

    Args:
        checkpoint_file: Path to the checkpoint file.

    Returns:
        The shard index to start from (0 if no checkpoint exists).
    """
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file) as f:
                return int(f.read().strip()) + 1
        except (ValueError, OSError):
            pass
    return 0


def save_checkpoint(checkpoint_file: str, shard_idx: int) -> None:
    """Save the last fully processed shard index to a checkpoint file.

    Args:
        checkpoint_file: Path to the checkpoint file.
        shard_idx: Index of the last completed shard.
    """
    with open(checkpoint_file, "w") as f:
        f.write(str(shard_idx))
