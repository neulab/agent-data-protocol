import json
import os

import tensorflow as tf
from tqdm import tqdm

# Load train episode IDs from split.json
split_file = "android_control/splits.json"
with open(split_file, "r") as f:
    split_data = json.load(f)
train_episode_ids = set(split_data["train"])

# Define the feature description
feature_description = {
    "episode_id": tf.io.FixedLenFeature([], tf.int64),
    "goal": tf.io.FixedLenFeature([], tf.string),
    "screenshots": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    "accessibility_trees": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    "screenshot_widths": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    "screenshot_heights": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    "actions": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    "step_instructions": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
}


# Function to parse the example
def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


# Function to process a single TFRecord file and extract the first 10 episodes
def process_tfrecord_file(tfrecord_file, writer, episodes_limit=3):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type="GZIP")
    parsed_dataset = raw_dataset.map(_parse_function)
    episode_count = 0

    for raw_record, parsed_record in zip(raw_dataset, parsed_dataset):
        record = {key: parsed_record[key].numpy() for key in feature_description.keys()}
        episode_id = int(record["episode_id"])

        # Skip records not in train_episode_ids
        if episode_id not in train_episode_ids:
            continue

        # Write the original raw record directly to the new TFRecord file
        writer.write(raw_record.numpy())
        episode_count += 1

        if episode_count >= episodes_limit:
            break


# Directory containing the TFRecord files
data_dir = "android_control"
output_file = "android_control/tf_record_sample"

# Get the list of TFRecord files
tfrecord_files = [
    os.path.join(data_dir, f) for f in os.listdir(data_dir) if not f.endswith(".json")
][0:1]

# Create a TFRecord writer
with tf.io.TFRecordWriter(output_file, options="GZIP") as writer:
    for tfrecord_file in tqdm(tfrecord_files, desc="Processing TFRecord files"):
        process_tfrecord_file(tfrecord_file, writer)

print("Sample TFRecord file created successfully.")
