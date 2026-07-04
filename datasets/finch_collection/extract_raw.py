import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

DATASET_ID = "minnesotanlp/Finch-Collection"


def parquet_files() -> list[str]:
    info = HfApi().dataset_info(DATASET_ID, files_metadata=False)
    return sorted(
        sibling.rfilename
        for sibling in info.siblings
        if sibling.rfilename.startswith("data/") and sibling.rfilename.endswith(".parquet")
    )


def iter_parquet_rows(path: Path, batch_size: int) -> Iterable[dict[str, Any]]:
    parquet_file = pq.ParquetFile(path)
    for row_group in range(parquet_file.num_row_groups):
        for batch in parquet_file.iter_batches(batch_size=batch_size, row_groups=[row_group]):
            for row in batch.to_pylist():
                yield row


def iter_rows(batch_size: int, limit: int | None) -> Iterable[dict[str, Any]]:
    emitted = 0
    for filename in parquet_files():
        path = Path(hf_hub_download(repo_id=DATASET_ID, repo_type="dataset", filename=filename))
        for row in iter_parquet_rows(path, batch_size):
            row["source_dataset"] = DATASET_ID
            row["source_file"] = filename
            row["row_index"] = emitted
            row["id"] = str(row.get("global_uid") or f"finch_collection_{emitted}")
            yield row
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream Finch Collection rows from Hugging Face Parquet shards."
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    try:
        for row in iter_rows(args.batch_size, args.limit):
            print(json.dumps(row, ensure_ascii=False), flush=True)
    except BrokenPipeError:
        sys.exit(0)


if __name__ == "__main__":
    main()
