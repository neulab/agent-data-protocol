import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

from schema_raw import SchemaRaw

from schema.action.api import ApiAction
from schema.action.message import MessageAction
from schema.observation.image import ImageObservation
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory

DOWNLOAD_INSTRUCTIONS = """
    # Please download the raw dumps first:
    cd datasets/weblinx/
    git clone https://huggingface.co/datasets/McGill-NLP/WebLINX-full
    cd WebLINX-full/
    git lfs pull --exclude="candidates/*,chat/*,data/*,**/bboxes/*,*.mp4,*.png"
"""

INTENT_MAP = {
    "load": "goto",
    "click": "click",
    "textInput": "type",
    "paste": "type",
    "scroll": "scroll",
    "submit": "submit",
    "change": "select",
}

WEBLINX_DUMP = Path(__file__).parent / "WebLINX-full"

intents_skipped = set()

# ── Axtree loading ──────────────────────────────────────────────────────────

# Cache: axtree_json_path -> (flattened_text, set_of_bids)
_axtree_cache: dict[str, tuple[str | None, set[str]]] = {}


def _load_and_flatten_axtree(
    browsergym_dir: Path,
    shortcode: str,
    page_file: str,
    filter_som: bool,
) -> tuple[str | None, set[str]]:
    """Load a pre-computed axtree JSON and flatten to text.

    Args:
        browsergym_dir: Root of weblinx-browsergym data (contains demonstrations/).
        shortcode: Demonstration shortcode (e.g. 'cptbbef').
        page_file: Page filename from step.state.page (e.g. 'page-2-0.html').
        filter_som: Whether to apply set-of-marks (interactive elements) filtering.

    Returns:
        Tuple of (axtree_text, set_of_browsergym_ids). Returns (None, set()) if
        the axtree file is not available.
    """
    json_name = page_file.replace(".html", ".json")
    axtree_path = browsergym_dir / "demonstrations" / shortcode / "axtrees" / json_name
    cache_key = str(axtree_path)

    if cache_key in _axtree_cache:
        return _axtree_cache[cache_key]

    if not axtree_path.exists():
        _axtree_cache[cache_key] = (None, set())
        return None, set()

    try:
        from browsergym.utils.obs import flatten_axtree_to_str
    except ImportError:
        print(
            "Error: browsergym-core is required for pre-computed axtree loading. "
            "Install with: pip install browsergym-core",
            file=sys.stderr,
        )
        _axtree_cache[cache_key] = (None, set())
        return None, set()

    try:

        with open(axtree_path) as f:
            axtree_json = json.load(f)

        # Load extra_element_properties for SoM filtering
        extra_props = None
        if filter_som:
            props_path = (
                browsergym_dir / "demonstrations" / shortcode
                / "extra_element_properties" / json_name
            )
            if props_path.exists():
                with open(props_path) as f:
                    extra_props = json.load(f)
            else:
                print(
                    f"Warning: SoM requested but extra_element_properties missing "
                    f"for {shortcode}/{json_name}",
                    file=sys.stderr,
                )

        text = flatten_axtree_to_str(
            axtree_json,
            extra_properties=extra_props,
            filter_som_only=filter_som and extra_props is not None,
            skip_generic=False,  # Must keep generic elements — 75% of action targets are <div>/<span>
        )

        # Extract all browsergym_ids from the JSON for BID validation
        bids = {
            node["browsergym_id"]
            for node in axtree_json.get("nodes", [])
            if node.get("browsergym_id")
        }

        _axtree_cache[cache_key] = (text, bids)
        return text, bids

    except Exception as e:
        print(f"Warning: failed to load axtree {axtree_path}: {e}", file=sys.stderr)
        _axtree_cache[cache_key] = (None, set())
        return None, set()


def convert_step(
    step: Any,
    shortcode: str,
    browsergym_dir: Path | None = None,
    filter_som: bool = False,
) -> list[Union[TextObservation, MessageAction, WebObservation, ApiAction]]:
    """Convert a step in the raw data to a list of standardized actions.

    Args:
        step: The step to convert (from SchemaRaw).
        shortcode: The shortcode of the demonstration.
        browsergym_dir: Path to weblinx-browsergym data. If None, falls back
            to reading HTML from WebLINX-full and setting axtree=None.
        filter_som: Whether to apply SoM filtering to axtrees.
    """
    if step.type == "chat":
        if step.speaker == "instructor":
            return [TextObservation(content=step.utterance, source="user")]
        elif step.speaker == "navigator":
            return [MessageAction(content=step.utterance, description=None)]
        else:
            print(f"Unknown speaker: {step.speaker}", file=sys.stderr)
    elif step.action["intent"] == "load":
        return [
            ApiAction(
                function=INTENT_MAP[step.action["intent"]],
                kwargs={"url": step.action["arguments"]["metadata"]["url"]},
            )
        ]
    elif step.action["intent"] in INTENT_MAP:
        args = step.action["arguments"]
        image_observation = None
        if step.state and step.state.screenshot:
            img_path = (
                (
                    WEBLINX_DUMP
                    / "demonstrations"
                    / shortcode
                    / "screenshots"
                    / step.state.screenshot
                )
                .relative_to(Path.cwd())
                .as_posix()
            )
            image_observation = ImageObservation(content=img_path, source="environment")

        # Load pre-computed axtree if browsergym data is available
        axtree_text = None
        page_file = step.state.page if step.state else None
        if browsergym_dir and page_file:
            axtree_text, _bids = _load_and_flatten_axtree(
                browsergym_dir, shortcode, page_file, filter_som,
            )

        if axtree_text is not None:
            # Use pre-computed axtree, drop HTML
            web_observation = WebObservation(
                html=None,
                axtree=axtree_text,
                url=args["metadata"]["url"],
                viewport_size=[
                    args["metadata"]["viewportWidth"],
                    args["metadata"]["viewportHeight"],
                ],
                image_observation=image_observation,
            )
        elif page_file:
            # Fallback: store raw HTML (original behavior)
            web_observation = WebObservation(
                html=(
                    WEBLINX_DUMP / "demonstrations" / shortcode / "pages" / page_file
                ).read_text(encoding="utf-8", errors="replace"),
                url=args["metadata"]["url"],
                viewport_size=[
                    args["metadata"]["viewportWidth"],
                    args["metadata"]["viewportHeight"],
                ],
                image_observation=image_observation,
                axtree=None,
            )
        else:
            # No page file available — observation with URL/screenshot only
            web_observation = WebObservation(
                html=None,
                axtree=None,
                url=args["metadata"]["url"],
                viewport_size=[
                    args["metadata"]["viewportWidth"],
                    args["metadata"]["viewportHeight"],
                ],
                image_observation=image_observation,
            )

        if step.action["intent"] == "scroll":
            return [
                web_observation,
                ApiAction(
                    function=INTENT_MAP[step.action["intent"]],
                    kwargs={"dx": args["scrollX"], "dy": args["scrollY"]},
                ),
            ]

        # Resolve element identifier: prefer BID over xpath
        _elid = args["element"]["attributes"].get("data-webtasks-id")
        if _elid and axtree_text is not None:
            # data-webtasks-id IS browsergym_id directly (verified)
            element_ref = {"bid": f'"{_elid}"'}
        elif _elid:
            # Have data-webtasks-id but no axtree — use xpath fallback
            element_ref = {"xpath": f"//*[@data-webtasks-id='{_elid}']"}
        else:
            # No data-webtasks-id — use raw xpath (~1% of actions)
            element_ref = {"xpath": args["element"]["xpath"]}

        if step.action["intent"] in ["click", "submit"]:
            return [
                web_observation,
                ApiAction(
                    function=INTENT_MAP[step.action["intent"]],
                    kwargs=element_ref,
                ),
            ]
        elif step.action["intent"] in ["textInput", "paste", "change"]:
            value_key = {
                "textInput": "text",
                "paste": "pasted",
                "change": "value",
            }
            value = args[value_key[step.action["intent"]]]
            return [
                web_observation,
                ApiAction(
                    function=INTENT_MAP[step.action["intent"]],
                    kwargs={**element_ref, "value": value},
                ),
            ]
        else:
            print(f"Unknown intent: {step.action['intent']}", file=sys.stderr)
    else:
        intents_skipped.add(step.action["intent"])
    return []


def process_data(
    raw_data_list: List[Dict],
    browsergym_dir: Path | None = None,
    filter_som: bool = False,
) -> List[Dict]:
    """Process a list of raw data into standardized trajectories.

    Args:
        raw_data_list: List of raw data dictionaries.
        browsergym_dir: Path to weblinx-browsergym data.
        filter_som: Whether to apply SoM filtering.

    Returns:
        List of standardized trajectory dictionaries.
    """
    standardized_trajectories = []

    for raw_data in raw_data_list:
        data = SchemaRaw(**raw_data)

        content: list = []
        for step in data.data:
            content.extend(convert_step(step, data.shortcode, browsergym_dir, filter_som))

        standardized_data = Trajectory(
            id=data.shortcode,
            content=content,
            details={
                "description": data.description,
                "tasks": ", ".join(data.tasks),
            },
        )
        standardized_trajectories.append(standardized_data.model_dump())

    return standardized_trajectories


def create_sample_std():
    """Create a sample standardized trajectory for testing purposes."""
    sample_trajectory = {
        "id": "sample-weblinx-trajectory",
        "content": [
            {
                "class_": "text_observation",
                "content": "I need help with booking a flight",
                "source": "user",
            },
            {
                "class_": "message_action",
                "content": "I'll help you book a flight. Let me navigate to a travel website.",
                "description": None,
            },
            {
                "class_": "api_action",
                "function": "goto",
                "kwargs": {"url": "https://example.com/flights"},
                "description": None,
            },
            {
                "class_": "web_observation",
                "html": "<html><body>Flight booking page</body></html>",
                "url": "https://example.com/flights",
                "viewport_size": [1024, 768],
                "image_observation": {
                    "class_": "image_observation",
                    "content": "datasets/weblinx/sample_screenshot.png",
                    "source": "environment",
                    "annotations": [],
                },
                "axtree": None,
            },
            {
                "class_": "api_action",
                "function": "type",
                "kwargs": {"bid": "\"example-bid\"", "value": "New York"},
                "description": None,
            },
        ],
    }
    return [sample_trajectory]


def process_single_data(
    raw_data: Dict,
    browsergym_dir: Path | None = None,
    filter_som: bool = False,
) -> Dict:
    """Process a single raw data into a standardized trajectory.

    Args:
        raw_data: Raw data dictionary.
        browsergym_dir: Path to weblinx-browsergym data.
        filter_som: Whether to apply SoM filtering.

    Returns:
        Standardized trajectory dictionary.
    """
    data = SchemaRaw(**raw_data)

    content: list = []
    for step in data.data:
        content.extend(convert_step(step, data.shortcode, browsergym_dir, filter_som))

    standardized_data = Trajectory(
        id=data.shortcode,
        content=content,
        details={
            "description": data.description,
            "tasks": ", ".join(data.tasks),
        },
    )
    return standardized_data.model_dump()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert raw weblinx data to standardized format.",
    )
    parser.add_argument(
        "--browsergym-dir",
        type=Path,
        default=None,
        help="Path to weblinx-browsergym data (contains demonstrations/ with "
             "pre-computed axtrees). Falls back to WEBLINX_BROWSERGYM_DIR env var, "
             "then to datasets/weblinx/browsergym-data/.",
    )
    parser.add_argument(
        "--filter-som",
        action="store_true",
        default=False,
        help="Apply set-of-marks (interactive elements only) filtering to axtrees. "
             "Requires extra_element_properties files in browsergym-dir.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Resolve browsergym data directory
    browsergym_dir = args.browsergym_dir
    if browsergym_dir is None:
        env_dir = os.environ.get("WEBLINX_BROWSERGYM_DIR")
        if env_dir:
            browsergym_dir = Path(env_dir)
        else:
            # Default: look for browsergym-data/ next to this script
            default_dir = Path(__file__).parent / "browsergym-data"
            if default_dir.is_dir():
                browsergym_dir = default_dir

    if browsergym_dir and browsergym_dir.is_dir():
        print(f"Using pre-computed axtrees from: {browsergym_dir}", file=sys.stderr)
    else:
        print(
            "No browsergym data found — falling back to raw HTML (axtree=None).",
            file=sys.stderr,
        )
        browsergym_dir = None

    # Check if WebLINX-full directory exists
    if not WEBLINX_DUMP.is_dir():
        print(
            f"Warning: {WEBLINX_DUMP} directory not found. Using sample data instead.",
            file=sys.stderr,
        )
        print(f"{DOWNLOAD_INSTRUCTIONS}", file=sys.stderr)
        # Create a sample standardized trajectory for testing
        standardized_trajectories = create_sample_std()

        # Print the standardized data as JSONL (one JSON object per line)
        for traj in standardized_trajectories:
            print(json.dumps(traj))
    else:
        # Process each line of input individually
        for line in sys.stdin:
            raw_data = json.loads(line)
            standardized_data = process_single_data(
                raw_data,
                browsergym_dir=browsergym_dir,
                filter_som=args.filter_som,
            )

            # Print the standardized data as JSON
            print(json.dumps(standardized_data))

        if intents_skipped:
            print("intents skipped: " + ", ".join(intents_skipped), file=sys.stderr)
