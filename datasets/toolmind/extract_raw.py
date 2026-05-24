import argparse
import json
import signal
import sys
import urllib.request
from typing import Iterator

signal.signal(signal.SIGPIPE, signal.SIG_DFL)

DATA_FILES = {
    "graphsyn": "https://huggingface.co/datasets/Nanbeige/ToolMind/resolve/main/graph_syn_datasets/graphsyn.jsonl",
    "apigen_mt": "https://huggingface.co/datasets/Nanbeige/ToolMind/resolve/main/open_datasets/APIGen-MT-5k-query.jsonl",
    "button_instruct": "https://huggingface.co/datasets/Nanbeige/ToolMind/resolve/main/open_datasets/BUTTONInstruct-query.jsonl",
    "toolace": "https://huggingface.co/datasets/Nanbeige/ToolMind/resolve/main/open_datasets/ToolACE-query.jsonl",
    "when2call": "https://huggingface.co/datasets/Nanbeige/ToolMind/resolve/main/open_datasets/When2Call-query.jsonl",
    "glaive": "https://huggingface.co/datasets/Nanbeige/ToolMind/resolve/main/open_datasets/glaive-function-calling-v2-query.jsonl",
    "tau_train": "https://huggingface.co/datasets/Nanbeige/ToolMind/resolve/main/open_datasets/tau-train-query.jsonl",
    "xlam": "https://huggingface.co/datasets/Nanbeige/ToolMind/resolve/main/open_datasets/xlam-function-calling-60k-query.jsonl",
}


def iter_rows(source: str, url: str) -> Iterator[dict]:
    with urllib.request.urlopen(url, timeout=120) as response:
        for row_index, line in enumerate(response):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("source_file", source)
            row.setdefault("row_index", row_index)
            row.setdefault("id", f"toolmind_{source}_{row_index}")
            yield row


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream ToolMind JSONL rows from Hugging Face.")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["graphsyn"],
        choices=sorted(DATA_FILES),
        help="ToolMind source files to stream, in order.",
    )
    args = parser.parse_args()

    try:
        for source in args.sources:
            for row in iter_rows(source, DATA_FILES[source]):
                print(json.dumps(row, ensure_ascii=False))
    except BrokenPipeError:
        sys.exit(0)


if __name__ == "__main__":
    main()
