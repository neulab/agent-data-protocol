import argparse
import glob
import os

import gymnasium as gym
import tqdm
from browsergym.utils.obs import flatten_axtree_to_str


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Directory containing HTML files.")
    parser.add_argument("output_dir", type=str, help="Directory to save the AXTree files")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    assert os.path.isdir(args.input_dir), f"{args.input_dir} does not exist or is not a directory."
    os.makedirs(args.output_dir, exist_ok=True)

    env = gym.make(
        "browsergym/openended",
        headless=True,
        task_kwargs={"start_url": "https://www.google.com"},
        wait_for_user_message=False,
    )
    obs, info = env.reset()

    html_files = glob.glob(os.path.join(args.input_dir, "*.html"))
    for file_path in tqdm.tqdm(html_files, desc="Processing HTML files"):
        file_path = os.path.abspath(file_path)
        obs, reward, terminated, truncated, info = env.step(f"goto('file://{file_path}')")
        with open(os.path.join(args.output_dir, os.path.basename(file_path) + ".axtree"), "w") as f:
            f.write(flatten_axtree_to_str(obs["axtree_object"]))
