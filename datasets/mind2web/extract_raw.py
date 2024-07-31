from datasets import load_dataset
import json
import globus_sdk
from globus_sdk.scopes import TransferScopes
import os
import subprocess
import tqdm
import time
import sys

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
            if entry['type'] == 'dir':
                dirs_to_process.append(os.path.join(current_dir, entry['name']))
            elif entry['type'] == 'file':
                if entry['name'].endswith('.zip'):
                    files.append(os.path.join(current_dir, entry['name']))
    
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
    with tqdm.tqdm(total=100, desc="Transfer Progress", unit='%') as pbar:
        while True:
            task = client.get_task(task_id)
            completion = task['subtasks_succeeded']/task['subtasks_total'] * 100
            pbar.n = completion
            pbar.refresh()
            if task['subtasks_succeeded'] == task['subtasks_total']:
                break
            time.sleep(5)  # wait for 5 seconds before checking again

globus_endpoint_local_id = sys.argv[1]
CLIENT_ID = 'f53b4edf-5f65-4c19-8fae-21a3f9f7a3d3'
auth_client = globus_sdk.NativeAppAuthClient(CLIENT_ID)

transfer_client = login_and_get_transfer_client()

source_collection_id = "32e6b738-a0b0-47f8-b475-26bf1c5ebf19"
dest_collection_id = globus_endpoint_local_id

task_data = globus_sdk.TransferData(
    source_endpoint=source_collection_id, destination_endpoint=dest_collection_id
)

source_directory = "/data/raw_dump/task/"
destination_directory = os.path.join(os.getcwd(), "raw_dump/")


files_to_transfer = list_files(transfer_client, source_collection_id, source_directory)

for file_path in files_to_transfer:
    relative_path = os.path.relpath(file_path, source_directory)
    destination_path = os.path.join(destination_directory, relative_path)
    task_data.add_item(file_path, destination_path)

try:
    task_id = do_submit(transfer_client)
except globus_sdk.TransferAPIError as err:
    if not err.info.consent_required:
        raise
    print(
        "Encountered a ConsentRequired error.\n"
        "You must login a second time to grant consents.\n\n"
    )
    transfer_client = login_and_get_transfer_client(
        scopes=err.info.consent_required.required_scopes
    )
    task_id = do_submit(transfer_client)

monitor_transfer(transfer_client, task_id)

dataset = load_dataset("osunlp/Mind2Web", split="train")

for sample in dataset:
    print(json.dumps(sample))
