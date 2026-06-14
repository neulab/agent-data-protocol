import json
from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent / "datasets" / "agenttuning_mind2web"


def test_agenttuning_mind2web_samples_preserve_provenance():
    samples = json.loads((DATASET_DIR / "sample_std.json").read_text())

    for sample in samples:
        assert sample["extra"]["source_dataset"] == "agenttuning_mind2web"
        assert sample["extra"]["raw"]["id"] == sample["trajectory_id"]
