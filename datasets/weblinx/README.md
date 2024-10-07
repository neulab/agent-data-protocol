
# WebLINX

[WebLINX](https://github.com/McGill-NLP/WebLINX) is a large-scale benchmark of 100K interactions across 2300 expert demonstrations of conversational web navigation, where a digital agent controls a web browser and follows user instructions to solve real-world tasks in a multi-turn dialogue fashion.


# Steps to Process

1. **Install Dependencies**
     ```sh
     pip install -r requirements.txt
     ```

2. **Extract raw data**
     ```sh
     cd datasets/weblinx/
     git clone https://huggingface.co/datasets/McGill-NLP/WebLINX-full
     cd WebLINX-full/
     git lfs pull --exclude="candidates/*,chat/*,data/*,**/bboxes/*,*.mp4,*.png"
     python extract_raw.py > weblinx_raw.jsonl
     ```

2. **Convert raw data to standardized format**
     ```sh
     cat weblinx_raw.jsonl | python raw_to_standardized.py > weblinx_standardized.jsonl
     ```
