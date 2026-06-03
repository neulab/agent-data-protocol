"""Normalize ATIF JSONL records while keeping ATIF as both input and output.

Despite the historical script name, this does not emit legacy ADP records. It
standardizes ATIF tool names/arguments before downstream SFT converters consume
the ATIF trajectory.
"""

from scripts.atif_to_std_common import main

if __name__ == "__main__":
    main(__file__)
