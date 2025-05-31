#!/usr/bin/python3

import json
import os
from typing import Any, Dict, List, Tuple

import certifi
import numpy as np
import tensorflow as tf

# import io
from PIL import Image, ImageDraw

credential_path = os.path.join(
    os.environ["HOME"], ".config/gcloud/application_default_credentials.json"
)
if not os.path.exists(credential_path):
    raise FileNotFoundError(
        f"Credential file not found at {credential_path}\n Please run `gcloud auth application-default login` to set up your credentials."
    )

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path
os.environ["CURL_CA_BUNDLE"] = certifi.where()

# Define dataset directories
dataset_directories = {
    "general": "gs://gresearch/android-in-the-wild/general/*",
    "google_apps": "gs://gresearch/android-in-the-wild/google_apps/*",
    "install": "gs://gresearch/android-in-the-wild/install/*",
    "single": "gs://gresearch/android-in-the-wild/single/*",
    "web_shopping": "gs://gresearch/android-in-the-wild/web_shopping/*",
}

# get the path that the script is in
script_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(script_dir, "screenshots")
os.makedirs(image_dir, exist_ok=True)


def parse_image_data(image_bytes, height, width, nb_channels) -> Image.Image:
    #  function parse_image_data {{{ #
    img_data: np.ndarray = np.frombuffer(
        image_bytes, dtype=np.uint8, count=height * width * nb_channels
    )
    img_data.shape = (height, width, nb_channels)
    img: Image.Image = Image.fromarray(img_data)
    return img
    #  }}} function parse_image_data #


# ACTION_TYPES = [ "dual-point gesture", "type"
# , "go_back", "go_home", "enter"
# , "task_complete", "task_impossible"
# ]
ACTION_TYPES = {
    3: "type",
    4: "dual-point gesture",
    5: "go_back",
    6: "go_home",
    7: "enter",
    10: "task_complete",
    11: "task_impossible",
}

for dataset_name, directory in dataset_directories.items():
    file_names = tf.io.gfile.glob(directory)

    dataset = tf.data.TFRecordDataset(file_names, compression_type="GZIP")
    json_list: List[Dict[str, Any]] = []
    for i, rcd in enumerate(dataset):
        if i >= 5:
            break

        example = tf.train.Example()
        example.ParseFromString(rcd.numpy())

        json_dict: Dict[str, Any] = {}
        for k, ftr in example.features.feature.items():
            data_type: str = ftr.WhichOneof("kind")
            json_dict[k] = list(getattr(ftr, data_type).value)

        # string data
        for k in [
            "device_type",
            "results/type_action",
            "episode_id",
            "current_activity",
            "goal_info",
        ]:
            json_dict[k] = json_dict[k][0].decode()

        # int
        for k in ["episode_length", "android_api_level", "step_id"]:
            json_dict[k] = json_dict[k][0]

        # others
        # not sure about the meaning of action_type, convert it according Sec. 3 in
        # the paper
        json_dict["results/action_type"] = ACTION_TYPES[json_dict["results/action_type"][0]]

        # (y, x, H, W)
        bboxes: np.ndarray = np.reshape(
            np.array(json_dict["image/ui_annotations_positions"]), (-1, 4)
        )
        json_dict["image/ui_annotations_positions"] = bboxes.tolist()
        json_dict["image/ui_annotations_ui_types"] = list(
            map(bytes.decode, json_dict["image/ui_annotations_ui_types"])
        )
        json_dict["image/ui_annotations_text"] = list(
            map(bytes.decode, json_dict["image/ui_annotations_text"])
        )

        # image
        json_dict["image/height"] = json_dict["image/height"][0]
        json_dict["image/width"] = json_dict["image/width"][0]
        json_dict["image/channels"] = json_dict["image/channels"][0]

        image_file_name: str = "{}/{:}-{:d}".format(
            image_dir, json_dict["episode_id"], json_dict["step_id"]
        )
        raw_screenshot: Image.Image = parse_image_data(
            json_dict["image/encoded"][0],
            json_dict["image/height"],
            json_dict["image/width"],
            json_dict["image/channels"],
        )
        raw_screenshot.save(image_file_name + ".png")

        drawer = ImageDraw.Draw(raw_screenshot, mode="RGB")
        for bb, lbl in zip(
            json_dict["image/ui_annotations_positions"], json_dict["image/ui_annotations_ui_types"]
        ):
            drawer.rectangle(
                [
                    json_dict["image/width"] * bb[1],
                    json_dict["image/height"] * bb[0],
                    json_dict["image/width"] * (bb[1] + bb[3]),
                    json_dict["image/height"] * (bb[0] + bb[2]),
                ],
                outline="red",
            )
            text_position: Tuple[int, int] = (
                json_dict["image/width"] * bb[1],
                json_dict["image/height"] * (bb[0] + bb[2]),
            )
            text_bbox: Tuple[int, int, int, int] = drawer.textbbox(text_position, lbl, anchor="lb")
            drawer.rectangle(text_bbox, fill="black")
            drawer.text(text_position, lbl, anchor="lb", fill="white")
        raw_screenshot.save(image_file_name + "-annotated.png")

        json_dict["image/encoded"] = image_file_name

        print(json.dumps(json_dict))
