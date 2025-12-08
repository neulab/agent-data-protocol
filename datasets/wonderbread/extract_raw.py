import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import zipfile

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Extract raw data from Wonderbread dataset")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory for dataset (default: datasets/wonderbread)",
    )
    return parser.parse_args()


args = parse_args()

# Root directory - configurable via argument, environment variable, or default
root = args.root if args.root else os.environ.get("WONDERBREAD_ROOT", "datasets/wonderbread")
zenodo_link = "https://zenodo.org/records/12671568/files/demos.zip?download=1"
source_file_name = root + "/" + os.path.basename(urllib.parse.urlparse(zenodo_link).path)
data_folder = source_file_name.split(".")[0]
extraction_complete_marker = data_folder + "/.extraction_complete"


def extract_sop(s: str) -> str:
    sop = []
    for line in s.split("\n"):
        if line and line[0].isdigit():
            sop.append(line)
    return "\n".join(sop)


if not os.path.exists(extraction_complete_marker):
    # Clean up partial extraction if it exists
    if os.path.exists(data_folder):
        print(f"Removing incomplete extraction: {data_folder}", file=sys.stderr)
        shutil.rmtree(data_folder)

    # Download the file (wget -c resumes partial downloads)
    subprocess.run(["wget", "-c", zenodo_link, "-O", source_file_name], check=True)

    # Unzip the file
    with zipfile.ZipFile(source_file_name, "r") as zip_ref:
        zip_ref.extractall(root)

    # Create completion marker
    with open(extraction_complete_marker, "w") as f:
        f.write("extraction complete\n")

    # Remove the zip file
    os.remove(source_file_name)
else:
    print(f"Using previously extracted data: {data_folder}", file=sys.stderr)

# enumerate the files
for task_stamp in tqdm(os.listdir(data_folder), desc="Processing tasks", file=sys.stderr):
    task_folder = f"{data_folder}/{task_stamp}"
    if task_stamp == ".DS_Store":
        continue

    # move screenshots to "./screenshots/task_stamp"
    screenshots_folder = f"{root}/screenshots/{task_stamp}"
    os.makedirs(screenshots_folder, exist_ok=True)
    for img in os.listdir(f"{task_folder}/screenshots"):
        os.rename(
            os.path.join(f"{task_folder}/screenshots", img),
            os.path.join(screenshots_folder, img),
        )

    if task_stamp[-3:] == "(1)":
        task_stamp = task_stamp[:-4]
    with open(f"{task_folder}/{task_stamp}.json") as f:
        data = json.load(f)
        wa_info = data.pop("webarena")
        task = wa_info["intent"]

    with open(f"{task_folder}/SOP - {task_stamp}.txt", "r") as f:
        sop = f.read()

    data["sop"] = extract_sop(sop)
    data["task_stamp"] = task_stamp
    data["task"] = task

    print(json.dumps(data))
