import json

from datasets import load_dataset

DATASET_NAME = "allenai/Dolci-Instruct-SFT-Tool-Use"


def main():
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    for sample in dataset:
        print(json.dumps(sample, ensure_ascii=False))


if __name__ == "__main__":
    main()
