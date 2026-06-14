import json
import sys
import urllib.error
import urllib.parse
import urllib.request

DATASET_ID = "open-thoughts/OpenThoughts-TB-dev"
HF_API_URL = f"https://huggingface.co/api/datasets/{DATASET_ID}?full=true"
HF_RAW_BASE = f"https://huggingface.co/datasets/{DATASET_ID}/raw/main"


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def fetch_text(path):
    quoted_path = urllib.parse.quote(path)
    url = f"{HF_RAW_BASE}/{quoted_path}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def get_source_files(metadata, task_id):
    prefix = f"{task_id}/"
    return sorted(
        sibling["rfilename"]
        for sibling in metadata["siblings"]
        if sibling["rfilename"].startswith(prefix)
    )


def iter_task_ids(metadata):
    task_ids = set()
    for sibling in metadata["siblings"]:
        filename = sibling["rfilename"]
        if filename.startswith(".") or "/" not in filename:
            continue
        task_id = filename.split("/", 1)[0]
        task_ids.add(task_id)
    for task_id in sorted(task_ids):
        source_files = get_source_files(metadata, task_id)
        if f"{task_id}/instruction.md" in source_files and f"{task_id}/task.toml" in source_files:
            yield task_id, source_files


def optional_file(task_id, relative_path):
    path = f"{task_id}/{relative_path}"
    try:
        return {"path": relative_path, "content": fetch_text(path)}
    except urllib.error.HTTPError:
        return None


def build_record(task_id, source_files):
    solution = optional_file(task_id, "solution/solve.sh")

    verification_files = []
    for relative_path in ["tests/test.sh", "tests/test_outputs.py"]:
        file_record = optional_file(task_id, relative_path)
        if file_record is not None:
            verification_files.append(file_record)

    return {
        "id": task_id,
        "instruction": fetch_text(f"{task_id}/instruction.md"),
        "task_toml": fetch_text(f"{task_id}/task.toml"),
        "solution": solution,
        "dockerfile": optional_file(task_id, "environment/Dockerfile"),
        "verification_files": verification_files,
        "source_files": source_files,
    }


def main():
    metadata = fetch_json(HF_API_URL)
    for task_id, source_files in iter_task_ids(metadata):
        record = build_record(task_id, source_files)
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.stderr.close()
