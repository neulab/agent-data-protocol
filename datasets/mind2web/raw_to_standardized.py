import json
import sys

from schema.action.api import ApiAction
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory
from datasets.mind2web.schema_raw import SchemaRaw, Action as RawAction

from playwright.sync_api import CDPSession, Page, ViewportSize

from .webarena_utils import (
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


def convert_step(step: RawAction) -> tuple[WebObservation, ApiAction]:
    web_observation = WebObservation(
        html=step.raw_html,
        # TODO: this should be added to the schema
        # https://github.com/neulab/agent-data-collection/issues/26
        image_observation=None,
        viewport_size=None,
        url=None,
    )

    # TODO: get the DOM element from `step.raw_html` here

    api_action = ApiAction(
        function=step.operation.op.lower(),
        kwargs={"value": step.operation.value} if step.operation.value else {},
        description=None,
    )
    return web_observation, api_action


for line in sys.stdin:

    raw_data = json.loads(line)
    data = SchemaRaw(**raw_data)

    content: list = [TextObservation(content=data.confirmed_task, source="user")]
    for action in data.actions:
        content.extend(convert_step(action))

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
