import json
from pathlib import Path


download_instructions = """
    cd datasets/weblinx/
    git clone https://huggingface.co/datasets/McGill-NLP/WebLINX-full
    cd WebLINX-full/
    git lfs pull --exclude="candidates/*,chat/*,data/*,**/bboxes/*,*.mp4,*.png"
"""

weblinx_dump = Path(__file__).parent / "WebLINX-full"
assert weblinx_dump.is_dir(), "Please download the dataset first: " + download_instructions

splits = json.loads((weblinx_dump / "splits.json").read_text())

for shortcode in splits["train"]:
    demo_dir = weblinx_dump / "demonstrations" / shortcode
    replay = json.loads((demo_dir / "replay.json").read_text())
    metadata = json.loads((demo_dir / "form.json").read_text())
    replay["shortcode"] = shortcode
    replay["description"] = metadata["description"]
    replay["tasks"] = metadata["tasks"]
    print(json.dumps(replay))