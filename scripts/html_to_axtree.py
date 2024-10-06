import os
import tqdm
import glob
import argparse
import gymnasium as gym
import browsergym.core  # register the openended task as a gym environment
from browsergym.utils.obs import flatten_axtree_to_str

class HTMLToAXTree:
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.env = gym.make(
            "browsergym/openended",
            headless=True,
            task_kwargs={"start_url": "https://www.google.com"},
            wait_for_user_message=False,
            tags_to_mark="all",
        )

    def build_axtree(self, html_content: str) -> str:
        temp_file = os.path.abspath(f'./temp_{self.dataset}.html')
        with open(temp_file, "w") as f:
            f.write(html_content)
        obs, info = self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(f"goto('file://{temp_file}')")
        os.remove(temp_file)

        return flatten_axtree_to_str(obs["axtree_object"])
    
if __name__ == "__main__":
    html_to_axtree = HTMLToAXTree()
    print(html_to_axtree.build_axtree("<html><body><h1>Hello World</h1></body></html>"))
