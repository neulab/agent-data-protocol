import importlib.util
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).parent.parent
    / "datasets"
    / "nvidia_SWE-Zero-openhands-trajectories"
    / "raw_to_standardized.py"
)


def load_converter_module():
    sys.path.insert(0, str(MODULE_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location("swe_zero_raw_to_standardized", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_normalize_tool_observation_strips_only_leading_prefix():
    module = load_converter_module()

    assert module.normalize_tool_observation("OBSERVATION:\nresult") == "result"
    assert (
        module.normalize_tool_observation("before\nOBSERVATION:\ninside output")
        == "before\nOBSERVATION:\ninside output"
    )
