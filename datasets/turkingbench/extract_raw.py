import os
import pandas as pd
import json


data_dir = os.path.join(os.path.dirname(__file__), "turking-bench", "tasks")


if __name__ == "__main__":
    for task in os.listdir(data_dir):
        with open(os.path.join(data_dir, task, "template.html"), "r") as f:
            template = f.read()
        data = pd.read_csv(os.path.join(data_dir, task, "batch.csv"), dtype=str)

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
