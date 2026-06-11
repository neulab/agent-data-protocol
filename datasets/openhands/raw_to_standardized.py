# ruff: noqa: E402, I001
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.raw_to_standardized_common import main


if __name__ == "__main__":
    main(__file__)
