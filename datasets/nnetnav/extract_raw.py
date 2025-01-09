#!/usr/bin/env python3
import json
from datasets import load_dataset

def main():
    ds = load_dataset("smurty/NNetNav-6k")
    
    for item in ds['train']:
        # Each item in the dataset is already in the format we want
        print(json.dumps(item))

if __name__ == "__main__":
    main()