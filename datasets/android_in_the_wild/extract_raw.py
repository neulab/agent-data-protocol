#!/usr/bin/python3

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import time

import certifi
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import errors_impl
from tqdm import tqdm

# import io
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract raw data from Android in the Wild dataset"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit samples per category (0=all, default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for screenshots (default: ./screenshots)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file path (default: stdout)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file",
    )
    parser.add_argument(
        "--skip-datasets",
        type=str,
        default="",
        help="Comma-separated list of dataset names to skip (e.g., 'general,google_apps,install')",
    )
    return parser.parse_args()


def load_processed_ids(output_path: str) -> set:
    """Load set of (episode_id, step_id) tuples from existing output."""
    processed = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    processed.add((record["episode_id"], record["step_id"]))
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Loaded {len(processed)} already-processed records", file=sys.stderr)
    return processed


args = parse_args()

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
image_dir = args.output_dir if args.output_dir else os.path.join(script_dir, "screenshots")
os.makedirs(image_dir, exist_ok=True)


def sanitize_filename(name: str, max_length: int = 150) -> str:
    """Replace filesystem-unsafe characters and truncate to safe length."""
    unsafe_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    result = name
    for char in unsafe_chars:
        result = result.replace(char, '_')

    # Truncate if too long, appending hash for uniqueness
    if len(result) > max_length:
        hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]
        result = result[:max_length - 9] + "_" + hash_suffix

    return result


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

# Initialize resume state and output file
processed_ids: set = set()
if args.resume and args.output:
    processed_ids = load_processed_ids(args.output)
    output_file = open(args.output, "a")
elif args.output:
    output_file = open(args.output, "w")
else:
    output_file = sys.stdout

skipped_count = 0

# Parse skip-datasets argument
skip_datasets = set(s.strip() for s in args.skip_datasets.split(',') if s.strip())

for dataset_name, directory in dataset_directories.items():
    if dataset_name in skip_datasets:
        print(f"Skipping {dataset_name} (in --skip-datasets)", file=sys.stderr)
        continue

    file_names = tf.io.gfile.glob(directory)

    dataset = tf.data.TFRecordDataset(file_names, compression_type="GZIP")
    json_list: List[Dict[str, Any]] = []
    try:
        for i, rcd in enumerate(tqdm(dataset, desc=f"Processing {dataset_name}", file=sys.stderr, mininterval=10)):
            if args.limit > 0 and i >= args.limit:
                break

            example = tf.train.Example()
            example.ParseFromString(rcd.numpy())

            json_dict: Dict[str, Any] = {}
            for k, ftr in example.features.feature.items():
                data_type: str = ftr.WhichOneof("kind")
                if data_type is None:
                    continue  # Skip features with no value set
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

            # Skip if already processed (for resume)
            if (json_dict["episode_id"], json_dict["step_id"]) in processed_ids:
                skipped_count += 1
                continue

            # others
            # not sure about the meaning of action_type, convert it according Sec. 3 in
            # the paper
            action_type_raw = json_dict["results/action_type"][0]
            json_dict["results/action_type"] = ACTION_TYPES.get(action_type_raw, f"unknown_{action_type_raw}")

            # (y, x, H, W) - UI annotations may be missing from some records
            ui_positions = json_dict.get("image/ui_annotations_positions", [])
            if ui_positions:
                bboxes: np.ndarray = np.reshape(np.array(ui_positions), (-1, 4))
                json_dict["image/ui_annotations_positions"] = bboxes.tolist()
            else:
                json_dict["image/ui_annotations_positions"] = []
            json_dict["image/ui_annotations_ui_types"] = list(
                map(bytes.decode, json_dict.get("image/ui_annotations_ui_types", []))
            )
            json_dict["image/ui_annotations_text"] = list(
                map(bytes.decode, json_dict.get("image/ui_annotations_text", []))
            )

            # image
            json_dict["image/height"] = json_dict["image/height"][0]
            json_dict["image/width"] = json_dict["image/width"][0]
            json_dict["image/channels"] = json_dict["image/channels"][0]

            safe_episode_id = sanitize_filename(json_dict["episode_id"])
            image_file_name: str = "{}/{:}-{:d}".format(
                image_dir, safe_episode_id, json_dict["step_id"]
            )

            # Only generate images if they don't already exist
            raw_image_path = image_file_name + ".png"
            annotated_image_path = image_file_name + "-annotated.png"

            if not os.path.exists(raw_image_path) or not os.path.exists(annotated_image_path):
                raw_screenshot: Image.Image = parse_image_data(
                    json_dict["image/encoded"][0],
                    json_dict["image/height"],
                    json_dict["image/width"],
                    json_dict["image/channels"],
                )
                raw_screenshot.save(raw_image_path)

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
                raw_screenshot.save(annotated_image_path)

            json_dict["image/encoded"] = image_file_name

            output_file.write(json.dumps(json_dict) + "\n")
            output_file.flush()
    except errors_impl.FailedPreconditionError as e:
        print(f"Network error while processing {dataset_name}: {e}", file=sys.stderr)
        print(f"Skipping to next dataset. Re-run with --resume to continue.", file=sys.stderr)
        time.sleep(5)
        continue

# Cleanup and summary
if args.output:
    output_file.close()

if skipped_count > 0:
    print(f"Skipped {skipped_count} already-processed records", file=sys.stderr)
