import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

DATASETS = {
    "v1-sft": "open-thoughts/OpenThoughts-Agent-v1-SFT",
    "sft-100k": "open-thoughts/OpenThoughts-Agent-SFT-100K",
}
DEFAULT_VARIANT = "sft-100k"


def parquet_files(dataset_id: str) -> list[str]:
    info = HfApi().dataset_info(dataset_id, files_metadata=False)
    return sorted(
        sibling.rfilename
        for sibling in info.siblings
        if sibling.rfilename.startswith("data/") and sibling.rfilename.endswith(".parquet")
    )


def row_id(row: dict[str, Any], index: int, variant: str) -> str:
    parts = [
        row.get("task"),
        row.get("run_id"),
        row.get("episode"),
        row.get("trial_name"),
    ]
    suffix = "_".join(str(part) for part in parts if part)
    return f"openthoughts_agent_{variant}_{suffix or index}"


def iter_parquet_rows(path: Path, batch_size: int) -> Iterable[dict[str, Any]]:
    parquet_file = pq.ParquetFile(path)
    for row_group in range(parquet_file.num_row_groups):
        for batch in parquet_file.iter_batches(batch_size=batch_size, row_groups=[row_group]):
            for row in batch.to_pylist():
                yield row


def iter_rows(variant: str, batch_size: int, limit: int | None) -> Iterable[dict[str, Any]]:
    dataset_id = DATASETS[variant]
    emitted = 0
    for filename in parquet_files(dataset_id):
        path = Path(hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=filename))
        for row in iter_parquet_rows(path, batch_size):
            row["source_dataset"] = dataset_id
            row["source_file"] = filename
            row["source_variant"] = variant
            row["row_index"] = emitted
            row["id"] = row_id(row, emitted, variant)
            yield row
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream OpenThoughts-Agent SFT rows from Hugging Face Parquet shards."
    )
    parser.add_argument(
        "--variant",
        choices=sorted(DATASETS),
        default=DEFAULT_VARIANT,
        help="OpenThoughts-Agent dataset variant to stream.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    try:
        for row in iter_rows(args.variant, args.batch_size, args.limit):
            print(json.dumps(row, ensure_ascii=False), flush=True)
    except BrokenPipeError:
        sys.exit(0)


if __name__ == "__main__":
    main()
