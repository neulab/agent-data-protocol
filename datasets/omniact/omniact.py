import json
import logging
import os
import io
import base64
from PIL import Image
import datasets

logger = logging.getLogger(__name__)

_URL = "https://huggingface.co/datasets/Writer/omniact/resolve/main/{}"


def image_to_base64(image_path):
    with Image.open(image_path) as image:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()


def fix_file_names(example):
    for key in example:
        if "/web/" in example[key]:
            if "screen_" in example[key] and key in ["image", "task"]:
                example[key] = example[key].replace("screen_", "screen")
            if key == "box":
                example[key] = example[key].replace("screen_", "screen")
                example[key] = example[key].replace(".json", "_boxes.json")

    return example


class OmniACTConfig(datasets.BuilderConfig):
    """BuilderConfig for OmniACT."""

    def __init__(self, include_image=False, **kwargs):
        super(OmniACTConfig, self).__init__(**kwargs)
        self.include_image = include_image


class OmniACT(datasets.GeneratorBasedBuilder):
    """OmniACT dataset."""

    BUILDER_CONFIGS = [
        OmniACTConfig(name="text", include_image=False),
        OmniACTConfig(name="image", include_image=True),
    ]

    DEFAULT_CONFIG_NAME = "text"

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "task": datasets.Value("string"),
                    "image": datasets.Value("string"),
                    "ocr": datasets.Sequence(
                        datasets.Features(
                            {
                                "text": datasets.Value("string"),
                                "value": datasets.Sequence(datasets.Value("float")),
                            }
                        )
                    ),
                    "color": datasets.Sequence(
                        datasets.Features(
                            {
                                "text": datasets.Value("string"),
                                "value": datasets.Sequence(
                                    datasets.Sequence(datasets.Value("float"))
                                ),
                            }
                        )
                    ),
                    "icon": datasets.Sequence(
                        datasets.Features(
                            {
                                "text": datasets.Value("string"),
                                "value": datasets.Sequence(
                                    datasets.Sequence(datasets.Value("float"))
                                ),
                            }
                        )
                    ),
                    "box": datasets.Sequence(
                        datasets.Features(
                            {
                                "top_left": datasets.Sequence(datasets.Value("float")),
                                "bottom_right": datasets.Sequence(
                                    datasets.Value("float")
                                ),
                                "label": datasets.Value("string"),
                            }
                        )
                    ),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": _URL.format("train.json"),
            "val": _URL.format("val.json"),
            "test": _URL.format("test.json"),
            "data": _URL.format("data.zip"),
            "ocr": _URL.format("ocr.zip"),
            "icon": _URL.format("detact_icon.zip"),
            "color": _URL.format("detact_color.zip"),
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "filepath": downloaded_files["train"],
                    "datapath": downloaded_files["data"],
                    "ocrpath": downloaded_files["ocr"],
                    "iconpath": downloaded_files["icon"],
                    "colorpath": downloaded_files["color"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": "val",
                    "filepath": downloaded_files["val"],
                    "datapath": downloaded_files["data"],
                    "ocrpath": downloaded_files["ocr"],
                    "iconpath": downloaded_files["icon"],
                    "colorpath": downloaded_files["color"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "filepath": downloaded_files["test"],
                    "datapath": downloaded_files["data"],
                    "ocrpath": downloaded_files["ocr"],
                    "iconpath": downloaded_files["icon"],
                    "colorpath": downloaded_files["color"],
                },
            ),
        ]

    def _generate_examples(
        self, split, filepath, datapath, ocrpath, iconpath, colorpath
    ):
        logger.info("Generating examples from = %s", filepath)
        with open(filepath, "r") as f:
            data = json.load(f)

        for example_id, doc in data.items():

            doc = fix_file_names(doc)

            with open(os.path.join(datapath, doc["task"]), "r") as f:
                task = f.read()

            if self.config.include_image:
                image = image_to_base64(os.path.join(datapath, doc["image"]))
            else:
                image = doc["image"]

            with open(os.path.join(ocrpath, doc["ocr"]), "r") as f:
                ocr_data = json.load(f)
                new_ocr_data = []
                for key, value in ocr_data.items():
                    new_ocr_data.append(
                        {
                            "text": key,
                            "value": value,
                        }
                    )
                ocr_data = new_ocr_data

            with open(os.path.join(iconpath, doc["icon"]), "r") as f:
                icon_data = json.load(f)
                new_icon_data = []
                for key, value in icon_data.items():
                    new_icon_data.append(
                        {
                            "text": key,
                            "value": value,
                        }
                    )
                icon_data = new_icon_data

            with open(os.path.join(datapath, doc["box"])) as f:
                box_data = json.load(f)
                new_box_data = []
                for key, value in box_data.items():
                    if value.get("valid", 1) == 1:
                        new_box_data.append(
                            {
                                "top_left": value["top_left"],
                                "bottom_right": value["bottom_right"],
                                "label": value["label"],
                            }
                        )
                box_data = new_box_data

            with open(os.path.join(colorpath, doc["color"])) as f:
                color_data = json.load(f)
                new_color_data = []
                for key, value in color_data.items():
                    new_color_data.append(
                        {
                            "text": key,
                            "value": value,
                        }
                    )
                color_data = new_color_data

            yield f"{split}_{example_id}", {
                "id": f"{split}_{example_id}",
                "task": task,
                "box": box_data,
                "image": image,
                "ocr": ocr_data,
                "icon": icon_data,
                "color": color_data,
            }
