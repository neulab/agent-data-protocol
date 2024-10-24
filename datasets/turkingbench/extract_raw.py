import os
import sys
import pandas as pd
import json
import subprocess


data_dir = os.path.join(os.path.dirname(__file__), "turking-bench", "tasks")
if not os.path.exists(data_dir):
    cmd = ["git", "clone", "--depth=1", "https://github.com/JHU-CLSP/turking-bench"]
    print(" ".join(cmd), file=sys.stderr)
    subprocess.run(
        cmd,
        cwd=os.path.dirname(__file__),
        stdout=sys.stderr,
        stderr=sys.stderr,
    )
assert os.path.isdir(data_dir)


if __name__ == "__main__":
    for task in os.listdir(data_dir):
        with open(os.path.join(data_dir, task, "template.html"), "r") as f:
            template = f.read()
        na_values = ["{}", "'{}'"]
        data = pd.read_csv(
            os.path.join(data_dir, task, "batch.csv"),
            dtype=str,
            na_filter=True,
            keep_default_na=False,
            na_values=na_values,
        )

        data["Task"] = task
        data["Template"] = template
        for index, row in data.iterrows():
            row = row.fillna("").to_dict()
            row["_id"] = row["Task"] + "__" + str(index)

            # TODO: handle this in SchemaRaw using @root_validator?
            # put all keys that start with "Answer." into a nested dict
            row["Answer"] = {
                k.split(".", 1)[1]: v for k, v in row.items() if k.startswith("Answer.")
            }
            # remove all keys that start with "Answer."
            row = {k: v for k, v in row.items() if not k.startswith("Answer.")}

            print(json.dumps(row))
