import json
import os
import sys
import urllib.error
import urllib.request

SOURCE_DATASET = os.environ.get("MIROVERSE_SOURCE_DATASET", "miromind-ai/MiroVerse-v0.1")
CONFIG_FILES = [
    ("MiroVerse-Voyager1.0", "jsonl_sft/MiroVerse-Voyager1.0.jsonl"),
    ("MiroVerse-MuSiQue", "jsonl_sft/MiroVerse-MuSiQue.jsonl"),
    ("MiroVerse-HotpotQA", "jsonl_sft/MiroVerse-HotpotQA.jsonl"),
    ("MiroVerse-WebWalkerQA-Silver", "jsonl_sft/MiroVerse-WebWalkerQA-Silver.jsonl"),
    ("MiroVerse-MegaScience", "jsonl_sft/MiroVerse-MegaScience.jsonl"),
    ("MiroVerse-TaskCraft", "jsonl_sft/MiroVerse-TaskCraft.jsonl"),
    ("MiroVerse-QA-Expert-Multi-Hop-V1.0", "jsonl_sft/MiroVerse-QA-Expert-Multi-Hop-V1.0.jsonl"),
    (
        "MiroVerse-OneGen-TrainDataset-MultiHopQA",
        "jsonl_sft/MiroVerse-OneGen-TrainDataset-MultiHopQA.jsonl",
    ),
    ("MiroVerse-2WikiMultihopQA", "jsonl_sft/MiroVerse-2WikiMultihopQA.jsonl"),
    ("MiroVerse-WikiTables", "jsonl_sft/MiroVerse-WikiTables.jsonl"),
    ("MiroVerse-WebShaper", "jsonl_sft/MiroVerse-WebShaper.jsonl"),
    ("MiroVerse-WebDancer", "jsonl_sft/MiroVerse-WebDancer.jsonl"),
]


def _selected_configs():
    requested = os.environ.get("MIROVERSE_CONFIGS")
    if not requested:
        return CONFIG_FILES
    wanted = {name.strip() for name in requested.split(",") if name.strip()}
    selected = [(name, path) for name, path in CONFIG_FILES if name in wanted]
    missing = wanted - {name for name, _ in selected}
    if missing:
        raise ValueError(f"Unknown MiroVerse configs: {sorted(missing)}")
    return selected


def _resolve_path(path):
    # The source repository stores SFT JSONL files under jsonl_sft/. Some public mirrors
    # flatten those files at the repository root; this keeps sample regeneration possible
    # without changing the default source dataset.
    if os.environ.get("MIROVERSE_FLAT_LAYOUT") == "1":
        return path.rsplit("/", 1)[-1]
    return path


def _open_hf_file(path):
    url = f"https://huggingface.co/datasets/{SOURCE_DATASET}/resolve/main/{_resolve_path(path)}"
    headers = {}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        return urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise RuntimeError(
                f"MiroVerse-v0.1 is gated on Hugging Face (HTTP {exc.code}). "
                "Accept the dataset terms at "
                "https://huggingface.co/datasets/miromind-ai/MiroVerse-v0.1 "
                "and provide an authorized HF_TOKEN, or set MIROVERSE_SOURCE_DATASET and "
                "MIROVERSE_FLAT_LAYOUT for a mirror with the same JSONL files."
            ) from exc
        raise


def iter_config_rows(config_name, path):
    with _open_hf_file(path) as response:
        for row_index, raw_line in enumerate(response):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line.decode("utf-8"))
            row.setdefault("split", config_name)
            row.setdefault("id", f"{config_name}-{row_index}")
            yield row


def main():
    max_per_config = os.environ.get("MIROVERSE_MAX_PER_CONFIG")
    max_per_config = int(max_per_config) if max_per_config else None
    for config_name, path in _selected_configs():
        for row_index, row in enumerate(iter_config_rows(config_name, path)):
            if max_per_config is not None and row_index >= max_per_config:
                break
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(exc, file=sys.stderr)
        raise
