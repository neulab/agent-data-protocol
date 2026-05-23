import json
import signal

from datasets import load_dataset

DATASET_NAME = "nebius/SWE-rebench-openhands-trajectories"

signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def main():
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    for item in dataset:
        if not item["resolved"]:
            continue
        try:
            print(json.dumps(item))
        except BrokenPipeError:
            return


if __name__ == "__main__":
    main()
