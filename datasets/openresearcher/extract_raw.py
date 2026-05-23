import argparse
import json
import sys
import urllib.parse
import urllib.request

DATASET = "OpenResearcher/OpenResearcher-Dataset"
DEFAULT_CONFIGS = [f"seed_{seed}" for seed in range(42, 58)]


def fetch_rows(config: str, split: str, offset: int, length: int) -> dict:
    params = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{params}"
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def iter_rows(
    configs: list[str],
    split: str,
    offset: int,
    limit: int | None,
    page_size: int,
    max_messages: int | None,
):
    emitted = 0
    for config in configs:
        current_offset = offset
        while True:
            if limit is not None:
                remaining = limit - emitted
                if remaining <= 0:
                    return
                length = min(page_size, remaining)
            else:
                length = page_size

            payload = fetch_rows(config, split, current_offset, length)
            rows = payload.get("rows", [])
            if not rows:
                break

            for item in rows:
                row = item["row"]
                if row.get("status") != "success":
                    continue
                if max_messages is not None and len(row.get("messages", [])) > max_messages:
                    continue
                row["config"] = config
                row["split"] = split
                row["row_idx"] = item.get("row_idx")
                yield row
                emitted += 1
                if limit is not None and emitted >= limit:
                    return

            current_offset += len(rows)
            total = payload.get("num_rows_total")
            if total is not None and current_offset >= total:
                break
            if len(rows) < length:
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract OpenResearcher raw trajectories.")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--split", default="train")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--page-size", type=int, default=10)
    parser.add_argument("--max-messages", type=int, default=None)
    args = parser.parse_args()

    try:
        for row in iter_rows(
            args.configs,
            args.split,
            args.offset,
            args.limit,
            args.page_size,
            args.max_messages,
        ):
            print(json.dumps(row, ensure_ascii=False))
    except BrokenPipeError:
        sys.exit(0)


if __name__ == "__main__":
    main()
