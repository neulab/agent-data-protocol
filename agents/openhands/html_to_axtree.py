import json
import os
import sys

import gymnasium as gym
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str
from lxml import etree


class HTMLToAXTree:
    def __init__(self, dataset: str):
        self.errors = []
        self.dataset = dataset
        self.env = gym.make(
            "browsergym/openended",
            headless=True,
            task_kwargs={"start_url": "https://www.google.com"},
            wait_for_user_message=False,
            tags_to_mark="all",
        )
        self.last_html = None
        self.last_xtree = None
        self.last_obs = None

    def build_axtree(self, id, html_content: str, chunk) -> str:
        self.last_html = html_content
        temp_dir = os.path.abspath("./temp/")
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        temp_file = os.path.abspath(f"{temp_dir}temp_{self.dataset}_{id}.html")

        with open(temp_file, "w") as f:
            f.write(html_content)

        obs, info = self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(f"page.goto('file://{temp_file}')")
        os.remove(temp_file)

        self.last_obs = obs
        self.last_xtree = flatten_axtree_to_str(obs["axtree_object"])

        return self.last_xtree

    def get_bid(self, id, x_path: str, chunk) -> str:
        html_string = flatten_dom_to_str(self.last_obs["dom_object"])
        tree = etree.HTML(html_string)
        try:
            if len(x_path) >= 2 and x_path[0] == x_path[-1] == '"':
                x_path = x_path[1:-1]
            element = tree.xpath(x_path)
            # print(x_path)
            # print(etree.tostring(element[0], pretty_print=True).decode("utf-8"))
            browsergym_id = element[0].get("bid")
            return browsergym_id
        except Exception as e:
            print("get_bid error:", e, file=sys.stderr)
            self.errors.append(
                {
                    "id": id,
                    "error": str(e),
                    "x_path": x_path,
                    "html_dom": html_string,
                    "raw_html": self.last_html,
                }
            )
            with open(
                f"./datasets/{self.dataset}/{self.dataset}_{chunk}_bid_errors.json", "w"
            ) as f:
                json.dump(self.errors, f, indent=4)
            return None


if __name__ == "__main__":
    html_to_axtree = HTMLToAXTree()
    print(html_to_axtree.build_axtree("<html><body><h1>Hello World</h1></body></html>"))
