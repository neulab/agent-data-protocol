import json
import signal

from datasets import load_dataset

signal.signal(signal.SIGPIPE, signal.SIG_DFL)

DATASET_NAME = "Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k"


def main():
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    for item in dataset:
        print(json.dumps(item))


if __name__ == "__main__":
    main()
