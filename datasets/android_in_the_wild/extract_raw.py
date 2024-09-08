#!/usr/bin/python3

import tensorflow as tf
import json
import os

from typing import Dict, List, Tuple
from typing import Any

#import io
from PIL import Image, ImageDraw
import numpy as np

file_names = [ "android-in-the-wild_general_general-00000-of-00321"
             , "android-in-the-wild_general_general-00001-of-00321"
             ]
output_file = "sample_raw.json"
os.makedirs("images", exist_ok=True)

def parse_image_data(image_bytes, height, width, nb_channels) -> Image.Image:
    #  function parse_image_data {{{ # 
    img_data: np.ndarray = np.frombuffer( image_bytes, dtype=np.uint8
                                        , count=height*width*nb_channels
                                        )
    img_data.shape = (height, width, nb_channels)
    img: Image.Image = Image.fromarray(img_data)
    return img
    #  }}} function parse_image_data # 

#ACTION_TYPES = [ "dual-point gesture", "type"
               #, "go_back", "go_home", "enter"
               #, "task_complete", "task_impossible"
               #]
ACTION_TYPES = { 3: "type", 4: "dual-point gesture"
               , 5: "go_back", 6: "go_home", 7: "enter"
               , 10: "task_complete", 11: "task_impossible"
               }

dataset = tf.data.TFRecordDataset(file_names, compression_type="GZIP")
json_list: List[Dict[str, Any]] = []
for i, rcd in enumerate(dataset):
    if i>=5:
        break

    example = tf.train.Example()
    example.ParseFromString(rcd.numpy())

    json_dict: Dict[str, Any] = {}
    for k, ftr in example.features.feature.items():
        data_type: str = ftr.WhichOneof("kind")
        json_dict[k] = list(getattr(ftr, data_type).value)

    # string data
    for k in ["device_type", "results/type_action", "episode_id", "current_activity", "goal_info"]:
        json_dict[k] = json_dict[k][0].decode()

    # int
    for k in ["episode_length", "android_api_level", "step_id"]:
        json_dict[k] = json_dict[k][0]

    # others
    # not sure about the meaning of action_type, convert it according Sec. 3 in
    # the paper
    json_dict["results/action_type"] = ACTION_TYPES[json_dict["results/action_type"][0]]

    # (y, x, H, W)
    bboxes: np.ndarray = np.reshape( np.array(json_dict["image/ui_annotations_positions"])
                                   , (-1, 4)
                                   )
    json_dict["image/ui_annotations_positions"] = bboxes.tolist()
    json_dict["image/ui_annotations_ui_types"] = list(map(bytes.decode, json_dict["image/ui_annotations_ui_types"]))
    json_dict["image/ui_annotations_text"] = list(map(bytes.decode, json_dict["image/ui_annotations_text"]))

    # image
    json_dict["image/height"] = json_dict["image/height"][0]
    json_dict["image/width"] = json_dict["image/width"][0]
    json_dict["image/channels"] = json_dict["image/channels"][0]

    image_file_name: str = "images/{:}-{:d}".format(json_dict["episode_id"], json_dict["step_id"])
    raw_screenshot: Image.Image =\
            parse_image_data( json_dict["image/encoded"][0], json_dict["image/height"]
                            , json_dict["image/width"], json_dict["image/channels"]
                            )
    raw_screenshot.save(image_file_name + ".png")

    drawer = ImageDraw.Draw(raw_screenshot, mode="RGB")
    for bb, lbl in zip( json_dict["image/ui_annotations_positions"]
                      , json_dict["image/ui_annotations_ui_types"]
                      ):
        drawer.rectangle( [ json_dict["image/width"]*bb[1], json_dict["image/height"]*bb[0]
                          , json_dict["image/width"]*(bb[1]+bb[3]), json_dict["image/height"]*(bb[0]+bb[2])
                          ]
                        , outline="red"
                        )
        text_position: Tuple[int, int] = (json_dict["image/width"]*bb[1], json_dict["image/height"]*(bb[0]+bb[2]))
        text_bbox: Tuple[int, int, int, int] = drawer.textbbox(text_position, lbl, anchor="lb")
        drawer.rectangle(text_bbox, fill="black")
        drawer.text(text_position, lbl, anchor="lb", fill="white")
    raw_screenshot.save(image_file_name + "-annotated.png")

    json_dict["image/encoded"] = image_file_name

    json_list.append(json_dict)

with open(output_file, "w") as f:
    json.dump(json_list, f, ensure_ascii=False, indent="\t")
