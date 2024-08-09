from datasets import load_dataset
import json
import globus_sdk
from globus_sdk.scopes import TransferScopes
import os
import tqdm
import time


def do_submit(client):
    task_doc = client.submit_transfer(task_data)
    task_id = task_doc["task_id"]
    return task_id


def list_files(client, endpoint_id, path):
    files = []
    dirs_to_process = [path]

    while dirs_to_process:
        # if len(files) > 10:
        #     return files
        current_dir = dirs_to_process.pop()
        response = client.operation_ls(endpoint_id, path=current_dir)

        for entry in response:
            if entry["type"] == "dir":
                dirs_to_process.append(os.path.join(current_dir, entry["name"]))
            elif entry["type"] == "file":
                if entry["name"].endswith(".zip"):
                    files.append(os.path.join(current_dir, entry["name"]))

    return files


def login_and_get_transfer_client(*, scopes=TransferScopes.all):
    auth_client.oauth2_start_flow(requested_scopes=scopes)
    authorize_url = auth_client.oauth2_get_authorize_url()
    print(f"Please go to this URL and login:\n\n{authorize_url}\n")
    auth_code = input("Please enter the code here: ").strip()
    tokens = auth_client.oauth2_exchange_code_for_tokens(auth_code)
    transfer_tokens = tokens.by_resource_server["transfer.api.globus.org"]

    return globus_sdk.TransferClient(
        authorizer=globus_sdk.AccessTokenAuthorizer(transfer_tokens["access_token"])
    )


def monitor_transfer(client, task_id):
    with tqdm.tqdm(total=100, desc="Transfer Progress", unit="%") as pbar:
        while True:
            task = client.get_task(task_id)
            completion = task["subtasks_succeeded"] / task["subtasks_total"] * 100
            pbar.n = completion
            pbar.refresh()
            if task["subtasks_succeeded"] == task["subtasks_total"]:
                break
            time.sleep(5)  # wait for 5 seconds before checking again

dataset = load_dataset("osunlp/Mind2Web", split="train")

for sample in dataset:
    print(json.dumps(sample))
