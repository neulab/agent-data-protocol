import json
import os
import subprocess
import urllib.parse
import zipfile

root = "datasets/wonderbread"
zenodo_link = "https://zenodo.org/records/12671568/files/demos.zip?download=1"
source_file_name = root + "/" + os.path.basename(urllib.parse.urlparse(zenodo_link).path)
data_folder = source_file_name.split(".")[0]


def extract_sop(s: str) -> str:
    sop = []
    for line in s.split("\n"):
        if line and line[0].isdigit():
            sop.append(line)
    return "\n".join(sop)


if not os.path.exists(data_folder):
    # download the file
    subprocess.run(["wget", "-c", zenodo_link, "-O", source_file_name], check=True)

    # unzip the file
    with zipfile.ZipFile(source_file_name, "r") as zip_ref:
        zip_ref.extractall(root)

    # remove the zip file
    os.remove(source_file_name)

# enumerate the files
for task_stamp in os.listdir(data_folder):
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
