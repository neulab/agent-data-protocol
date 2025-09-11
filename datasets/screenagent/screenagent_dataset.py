import glob
import json
import os
from datetime import datetime

from datasets.features import Features, Value
from datasets.features import Image as hf_Image
from PIL import Image

import datasets
from datasets import DatasetInfo, GeneratorBasedBuilder, Sequence

_HOME_PAGE = "https://github.com/niuzaisheng/ScreenAgent"
_CITATION = """@article{niu2024screenagent,
      title={ScreenAgent: A Vision Language Model-driven Computer Control Agent},
      author={Runliang Niu and Jindong Li and Shiqi Wang and Yali Fu and Xiyu Hu and Xueyuan Leng and He Kong and Yi Chang and Qi Wang},
      year={2024},
      eprint={2402.07945},
      archivePrefix={arXiv},
      primaryClass={cs.HC}
}"""
_DESCRIPTION = "ScreenAgent: A Computer Control Agent Driven by Visual Language Large Model"
_URL = "https://codeload.github.com/niuzaisheng/ScreenAgent/zip/refs/heads/main"


class ScreenAgentDataset(GeneratorBasedBuilder):
    def _info(self):
        features = Features(
            {
                "data": Sequence(
                    {
                        "task_prompt": Value("string"),
                        "send_prompt": Value("string"),
                        "current_task": Value("string"),
                        "LLM_response": Value("string"),
                        "LLM_response_editer": Value("string"),
                        "video_height": Value("int32"),
                        "video_width": Value("int32"),
                        "saved_image_name": Value("string"),
                        "session_id": Value("string"),
                        "task_prompt_en": Value("string"),
                        "task_prompt_zh": Value("string"),
                        "send_prompt_en": Value("string"),
                        "send_prompt_zh": Value("string"),
                        "actions": Value("string"),
                        "LLM_response_editer_en": Value("string"),
                        "LLM_response_editer_zh": Value("string"),
                        "screenshot": hf_Image(),
                    }
                )
            }
        )
        return DatasetInfo(
            features=features,
            supervised_keys=None,
            homepage=_HOME_PAGE,
            citation=_CITATION,
            description=_DESCRIPTION,
        )

    def _split_generators(self, dl_manager):
        extracted_path = dl_manager.download_and_extract(_URL)
        train_data_dir = os.path.join(
            extracted_path, "ScreenAgent-main", "data", "ScreenAgent", "train"
        )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": train_data_dir},
            )
        ]

    def _generate_examples(self, data_dir):
        for session_id in os.listdir(data_dir):
            session_path = os.path.join(data_dir, session_id)

            step_paths = glob.glob(os.path.join(session_path, "*.json"))
            step_paths.sort(
                key=lambda x: datetime.strptime(
                    os.path.basename(x).split("_")[0] + "_" + os.path.basename(x).split("_")[1],
                    "%Y-%m-%d_%H-%M-%S-%f",
                )
            )

            traj = []
            for step_path in step_paths:
                if "neg" in os.path.split(step_path)[-1]:
                    continue

                with open(step_path, "r") as f:
                    one_step = json.load(f)
                    one_step["session_id"] = session_id
                    one_step["actions"] = json.dumps(one_step["actions"], indent=4)

                    if "current_task" not in one_step:
                        one_step["current_task"] = "N/A"

                    image_path = os.path.join(
                        data_dir, session_id, "images", one_step["saved_image_name"]
                    )
                    one_step["screenshot"] = Image.open(image_path).convert("RGB")

                    traj.append(one_step)

            yield session_id, {"data": traj}
