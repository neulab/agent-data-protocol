# Android in the Wild

This dataset is a collection of Android apps and their interactions, collected
from the wild. [arXiv paper](https://arxiv.org/abs/2307.10088)

## Setup

Before running `extract_raw.py`, you need to set up your Google Cloud
credentials. Please run `gcloud auth application-default login` to set up your
credentials.

## Data Location & Statistics

|     Name    |  # Episodes |   # Examples  | # Unique Prompts |                                                                  Data location                                                                 |
|:-----------:|:-----------:|:-------------:|:----------------:|:----------------------------------------------------------------------------------------------------------------------------------------------:|
|  GoogleApps |   625,542   |   4,903,601   |        306       |  [gs://gresearch/android-in-the-wild/google_apps](https://console.cloud.google.com/storage/browser/gresearch/android-in-the-wild/google_apps)  |
|   Install   |    25,760   |    250,058    |        688       |      [gs://gresearch/android-in-the-wild/install](https://console.cloud.google.com/storage/browser/gresearch/android-in-the-wild/install)      |
| WebShopping |    28,061   |    365,253    |      13,473      | [gs://gresearch/android-in-the-wild/web_shopping](https://console.cloud.google.com/storage/browser/gresearch/android-in-the-wild/web_shopping) |
|   General   |    9,476    |     85,413    |        545       |      [gs://gresearch/android-in-the-wild/general](https://console.cloud.google.com/storage/browser/gresearch/android-in-the-wild/general)      |
|    Single   |    26,303   |     85,668    |      15,366      |       [gs://gresearch/android-in-the-wild/single](https://console.cloud.google.com/storage/browser/gresearch/android-in-the-wild/single)       |
|  **Total**  | **715,142** | **5,689,993** |    **30,378**    |              [gs://gresearch/android-in-the-wild](https://console.cloud.google.com/storage/browser/gresearch/android-in-the-wild/)             |

## Data Format

Each datapoint is stored as a [TFRecord
file](https://www.tensorflow.org/tutorials/load_data/tfrecord#reading_a_tfrecord_file_2)
with compression type `'GZIP'` with the following fields:

*   `android_api_level`: the Android API level of the emulator the episode was
    collected from
*   `current_activity`: the name of the activity running when the example was
    collected
*   `device_type`: the device type of the emulator the episode was collected
    from, mostly Pixel devices with one custom device image
*   `episode_id`: the unique identifier for the episode the example is from
*   `episode_length`: the overall number of steps in the episode
*   `goal_info`: the natural language instruction the episode is demonstrating
*   `image/channels`, `image/height`, `image/width`: the number of channels,
    height, and width of the screenshot
*   `image/encoded`: the encoded screenshot
*   `image/ui_annotations_positions`: a flattened array of coordinates
    representing the bounding boxes of the UI annotations; the coordinates are
    in (y, x, height, width) format and the length of this array is `4 *
    num_elements`
*   `image/ui_annotations_text`: the OCR-detected text associated with the UI
    element
*   `image/ui_annotations_ui_types`: the type of UI element for each
    annotation, can be an icon or just text
*   `results/action_type`: the type of the predicted action (see 'Action space'
    for more details)
*   `results/type_action`: if the action is a `type` then this holds the text
    string that was typed
*   `results/yx_touch`, `results/yx_lift`: the (y, x) coordinates for the touch
    and lift point of a dual point action
*   `step_id`: the example's zero-indexed step number within the episode (i.e.
    if `step_id` is 2, then this is the third step of the episode)
