import tempfile
import urllib.request
import os

ORIG_FILE_LOC = "http://phontron.com/data/od-trajectories-20240701.jsonl.gz"

with tempfile.TemporaryDirectory() as tmpdirname:
    urllib.request.urlretrieve(ORIG_FILE_LOC, f"{tmpdirname}/data.jsonl.gz")
    # Unzip the file
    os.system(f"gunzip {tmpdirname}/data.jsonl.gz")
    with open(f"{tmpdirname}/data.jsonl", "r") as f:
        for line in f:
            print(line.strip())
