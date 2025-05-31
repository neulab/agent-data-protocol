import os
import subprocess
import zipfile

root = "datasets/webarena_successful"
# download the trajectory file
if not os.path.exists(f"{root}/trajectories.jsonl"):
    source_id = "1tvnaklsdSLx4Sp9Uc1spopcFpLktStO8"
    subprocess.run(["gdown", source_id, "-O", f"{root}/trajectories.jsonl"], check=True)

# download screenshot files
if not os.path.exists(f"{root}/screenshots"):
    source_id = "1TNfhApmiEIxiOcUqi4duvVWBaH5_m3By"
    subprocess.run(["gdown", source_id, "-O", f"{root}/screenshots.zip"], check=True)
    # unzip the file
    with zipfile.ZipFile(f"{root}/screenshots.zip", "r") as zip_ref:
        zip_ref.extractall(root)
    # rename images to screenshots
    os.rename(f"{root}/images", f"{root}/screenshots")
    os.remove(f"{root}/screenshots.zip")

with open(f"{root}/trajectories.jsonl", "r") as f:
    for line in f:
        line = line.replace("demo_trajs/images", f"{root}/screenshots")
        print(line.strip())
