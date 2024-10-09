import argparse
import json
import sys
import os

from schema.action.api import ApiAction
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory
from schema_raw import SchemaRaw, Action as RawAction
from bs4 import BeautifulSoup
from lxml import etree
import collections
import re
import tqdm

from playwright.sync_api import CDPSession, Page, ViewportSize, sync_playwright

from webarena_utils import (
    AccessibilityTree,
    AccessibilityTreeNode,
    BrowserConfig,
    BrowserInfo,
    DOMNode,
    DOMTree,
    Observation,
    png_bytes_to_numpy,
)


def fetch_browser_info(
    self,
    page: Page,
    client: CDPSession,
) -> BrowserInfo:
    # extract domtree
    tree = client.send(
        "DOMSnapshot.captureSnapshot",
        {
            "computedStyles": [],
            "includeDOMRects": True,
            "includePaintOrder": True,
        },
    )

    # calibrate the bounds, in some cases, the bounds are scaled somehow
    bounds = tree["documents"][0]["layout"]["bounds"]
    b = bounds[0]
    n = b[2] / self.viewport_size["width"]
    bounds = [[x / n for x in bound] for bound in bounds]
    tree["documents"][0]["layout"]["bounds"] = bounds

    # extract browser info
    win_top_bound = page.evaluate("window.pageYOffset")
    win_left_bound = page.evaluate("window.pageXOffset")
    win_width = page.evaluate("window.screen.width")
    win_height = page.evaluate("window.screen.height")
    win_right_bound = win_left_bound + win_width
    win_lower_bound = win_top_bound + win_height
    device_pixel_ratio = page.evaluate("window.devicePixelRatio")
    assert device_pixel_ratio == 1.0, "devicePixelRatio is not 1.0"

    config: BrowserConfig = {
        "win_top_bound": win_top_bound,
        "win_left_bound": win_left_bound,
        "win_width": win_width,
        "win_height": win_height,
        "win_right_bound": win_right_bound,
        "win_lower_bound": win_lower_bound,
        "device_pixel_ratio": device_pixel_ratio,
    }

    # assert len(tree['documents']) == 1, "More than one document in the DOM tree"
    info: BrowserInfo = {"DOMTree": tree, "config": config}
    self.d_tree = tree
    return info


def get_bounding_client_rect(client: CDPSession, backend_node_id: str):
    try:
        remote_object = client.send(
            "DOM.resolveNode", {"backendNodeId": int(backend_node_id)}
        )
        remote_object_id = remote_object["object"]["objectId"]
        response = client.send(
            "Runtime.callFunctionOn",
            {
                "objectId": remote_object_id,
                "functionDeclaration": """
                        function() {
                            if (this.nodeType == 3) {
                                var range = document.createRange();
                                range.selectNode(this);
                                var rect = range.getBoundingClientRect().toJSON();
                                range.detach();
                                return rect;
                            } else {
                                return this.getBoundingClientRect().toJSON();
                            }
                        }
                    """,
                "returnByValue": True,
            },
        )
        return response
    except Exception as e:
        return {"result": {"subtype": "error"}}


def process_trace(trace_file, page, record):
    info_mapping = {}
    page.goto("https://trace.playwright.dev/")

    with page.expect_file_chooser() as fc_info:
        page.get_by_role("button", name="Select file(s)").click()

    file_chooser = fc_info.value

    file_chooser.set_files(trace_file)
    trace_file = trace_file.split("/")[-2]
    # print(trace_file)
    action_mapping = collections.defaultdict(list)
    action_uids = []
    page.locator(".action-title").first.wait_for(timeout=50000)

    for idx in range(page.locator(".action-title").count()):
        action = page.locator(".action-title").nth(idx)
        action_repr = action.text_content()
        if action_repr.startswith("Keyboard.type"):
            action_uid = [action_uid]
        else:
            action_uid = re.findall(r"get_by_test_id\(\"(.+?)\"\)", action_repr)
        if (
            action_repr.startswith("Locator.count")
            or action_repr.startswith("Locator.all")
            or len(action_uid) == 0
        ):
            continue

        action_uid = action_uid[0]
        if action_uid not in action_mapping:
            action_uids.append(action_uid)
        action_mapping[action_uid].append(action)

    # print(action_uids)

    for action_uid in action_uids:
        for action_idx in range(len(action_mapping[action_uid])):
            if os.path.exists(record):
                with open(record, "r") as file:
                    content = file.read()
                    if (
                        trace_file + "-" + action_uid + "-" + f"{action_idx:03d}"
                    ) in content:
                        continue
            action_seq = action_mapping[action_uid]
            action_seq[action_idx].click()

            # before
            page.locator('div.tabbed-pane-tab-label:text("Before")').click()
            with page.expect_popup() as snapshot_popup:
                page.locator("button.link-external").click()
                # snapshot is the playwright page
                snapshot = snapshot_popup.value
                cdp_client = snapshot.context.new_cdp_session(snapshot)
                snapshot.wait_for_load_state("domcontentloaded")

                snapshot.evaluate(
                    """()=>{
                    const highlight_element = document.getElementById("x-pw-highlight-box");
                    if (highlight_element !== null) {highlight_element.style.display = "none";}
                }"""
                )
                tree = cdp_client.send(
                    "DOMSnapshot.captureSnapshot",
                    {
                        "computedStyles": [],
                        "includeDOMRects": True,
                        "includePaintOrder": True,
                    },
                )
                document = tree["documents"][0]
                strings = tree["strings"]
                if "data-pw-testid-buckeye" not in strings:
                    print('"data-pw-testid-buckeye" not in strings')
                    continue
                tgt_idx = strings.index("data-pw-testid-buckeye")
                nodes = document["nodes"]
                backend_node_ids = nodes["backendNodeId"]
                node_names = nodes["nodeName"]
                node_types = nodes["nodeType"]
                attributes = nodes["attributes"]
                backend_node_id = -1
                for idx in range(len(node_names)):
                    if tgt_idx in attributes[idx]:
                        action_uid_idx = attributes[idx][
                            attributes[idx].index(tgt_idx) + 1
                        ]
                        if strings[action_uid_idx] == action_uid:
                            backend_node_id = backend_node_ids[idx]
                            break
                if backend_node_id != -1:
                    # print(action_uid, backend_node_id)
                    # print('trace_file',trace_file)
                    info_mapping[trace_file + "-" + action_uid] = backend_node_id
                cdp_client.detach()
                snapshot.close()
    return info_mapping


def convert_step(step: RawAction, info_mapping: dict, annotation_id) -> tuple[WebObservation, ApiAction]:
    
    soup = BeautifulSoup(step.raw_html, 'html.parser')
    elements_with_attribute = soup.find_all(attrs={"data_pw_testid_buckeye": True})
    for element in elements_with_attribute:
        del element['data_pw_testid_buckeye']
    raw_html_no_label = soup.prettify()

    web_observation = WebObservation(
        html=raw_html_no_label,
        # TODO: this should be added to the schema
        # https://github.com/neulab/agent-data-collection/issues/26
        image_observation=None,
        viewport_size=None,
        url=None,
    )

    # TODO: get the DOM element from `step.raw_html` here
    # use info_mapping[action_uid] to retrieve node's attributes
    label_xpath = f"//*[@data_pw_testid_buckeye='{step.action_uid}']"
    tree = etree.HTML(step.raw_html)
    elements = tree.xpath(label_xpath)
    backend_node_id = elements[0].get("backend_node_id")

    dom_nodeid = info_mapping.get(f'{annotation_id}-{step.action_uid}', 'not found')
    if dom_nodeid == 'not found' and backend_node_id:
        dom_nodeid = backend_node_id
    xpath = f"//*[@backend_node_id='{dom_nodeid}']"

    api_action = ApiAction(
        function=step.operation.op.lower(),
        kwargs={"value": step.operation.value, "xpath": xpath} if step.operation.value else {"xpath": xpath},
        description=None,
    )
    return web_observation, api_action


if __name__ == "__main__":

    # add argparse to get location of raw_dump
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dump", type=str)
    args = parser.parse_args()

    trace_files = []
    info_mapping = dict()
    if args.raw_dump:
        print("ERROR: raw_dump functionality is not fully implemented yet", sys.stderr)
        sys.exit(1)
    if args.raw_dump and os.path.isdir(args.raw_dump):
        for root, dirs, files in os.walk(args.raw_dump):
            for file in files:
                if file == "trace.zip":
                    trace_file = os.path.join(root, file)
                    trace_files.append(trace_file)

        with sync_playwright() as p:
            p.selectors.set_test_id_attribute("data-pw-testid-buckeye")
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1280, "height": 1080})

            for trace_file in tqdm.tqdm(trace_files):
                success = False
                page = context.new_page()
                record_fn = "./record"
                if not os.path.exists(record_fn):
                    with open(record_fn, "w") as file:
                        pass
                try:
                    trace_info_mapping = process_trace(
                        trace_file,
                        page,
                        record=record_fn,
                    )
                    info_mapping.update(trace_info_mapping)

                    success = True
                except Exception as e:
                    print(e)
                page.close()
                if not success:
                    print(f"Failed to process {trace_file}")
            browser.close()


    for line in sys.stdin:

        raw_data = json.loads(line)
        data = SchemaRaw(**raw_data)

        content: list = [
            TextObservation(
                content=data.confirmed_task, source="user"
            )
        ]
        for action in data.actions:
            content.extend(convert_step(action,info_mapping,data.annotation_id))

        standardized_data = Trajectory(
            id=data.annotation_id,
            content=content,
            details={
                "website": data.website,
                "domain": data.domain,
                "confirmed_task": data.confirmed_task,
                "subdomain": data.subdomain,
            },
        )
        # Print the standardized data
        print(standardized_data.model_dump_json())
