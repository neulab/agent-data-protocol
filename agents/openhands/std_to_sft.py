import argparse
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path

from PIL import Image

from tqdm import tqdm

from agents.openhands.api import (
    browser_default_apis,
    get_api_tool_description,
    get_language_descriptions,
    openhands_default_tools,
)
from agents.openhands.system_prompt.system import get_system_message
from agents.openhands.system_prompt.user import get_web_user_message
from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.image import ImageObservation
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory
from scripts.html_to_axtree import HTMLToAXTree

dataset = os.getenv("MY_DATASET")
assert dataset, "Please set the environment variable MY_DATASET"

# Lazy-initialized: browser is only started when the first web observation
# needs axtree generation.  Non-web datasets never pay the startup cost.
_generate_axtree = None


def get_axtree_generator(filter_som: bool = False,
                         filter_visible: bool = False) -> HTMLToAXTree:
    """Return the shared HTMLToAXTree instance, creating it on first call.

    Filter flags are fixed at creation time. Subsequent calls with different
    flag values will raise AssertionError to prevent silent misconfiguration.

    Args:
        filter_som: Only include set-of-marks (interactive) elements.
        filter_visible: Only include visible elements (visibility >= 0.5).
    """
    global _generate_axtree
    if _generate_axtree is None:
        _generate_axtree = HTMLToAXTree(
            dataset, filter_som_only=filter_som, filter_visible_only=filter_visible,
        )
    elif isinstance(_generate_axtree, HTMLToAXTree):
        assert _generate_axtree.filter_som_only == filter_som, \
            f"Cannot change filter_som after init ({_generate_axtree.filter_som_only} != {filter_som})"
        assert _generate_axtree.filter_visible_only == filter_visible, \
            f"Cannot change filter_visible after init ({_generate_axtree.filter_visible_only} != {filter_visible})"
    return _generate_axtree

action_function = {"python": "execute_ipython_cell", "bash": "execute_bash", "web": "browser"}
function_args = {"execute_ipython_cell": "code", "execute_bash": "command", "browser": "code"}


def round_sigfigs(value, n: int):
    """Round a numeric value to n significant figures. Non-numeric values pass through.

    Args:
        value: The value to round. Non-float values are returned unchanged.
        n: Number of significant figures.

    Returns:
        Rounded float, or the original value if not a float.
    """
    if not isinstance(value, float):
        return value
    if value == 0:
        return 0.0
    return round(value, -int(math.floor(math.log10(abs(value)))) + (n - 1))


def estimate_text_tokens(text: str) -> int:
    """Estimate token count using three character classes.

    Qwen-family tokenizers handle different character types with very
    different efficiency, measured via analyze_token_efficiency.py:
    - Digits: ~1.0 chars/tok (numbers, coordinates, prices)
    - Punctuation/spaces/special: ~1.5 chars/tok (parens, commas, URLs)
    - Letters (alpha): ~4.0 chars/tok (words, dotted identifiers)
    """
    digits = 0
    alpha = 0
    other = 0
    for c in text:
        if c.isdigit():
            digits += 1
        elif c.isalpha():
            alpha += 1
        else:
            other += 1
    return int(digits / 1.0 + other / 1.5 + alpha / 4.0)


# Cache the system prompt and its estimated token count (constant across all records)
_SYSTEM_PROMPT = get_system_message()
_SYSTEM_PROMPT_TOKENS = estimate_text_tokens(_SYSTEM_PROMPT)

# Default per-axtree character cap. 30k chars ≈ 10k tokens. Preserves most normal
# axtrees while capping outliers that can reach 300k+ chars on complex websites.
DEFAULT_MAX_AXTREE_CHARS = 30000


class ConversionStats:
    """Collect stats during SFT conversion for summary reporting."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.trajectories_processed = 0
        self.trajectories_failed = 0
        self.truncations = 0
        self.real_bids_found = 0
        self.placeholder_bids_generated = 0
        self.first_role_fixes = 0
        self.message_merges = 0
        self.finish_appends = 0
        self.total_messages_before_merge = 0
        self.total_messages_after_merge = 0
        self.axtrees_truncated = 0
        self.early_terminations = 0
        self.events_skipped = 0
        self.samples_dropped_over_budget = 0
        self.bids_referenced_in_actions = 0
        self.bids_found_in_axtree = 0
        self.bids_missing_from_axtree = 0

    def print_summary(self):
        """Print summary stats to stderr."""
        print(f"\n[{dataset}] Conversion Summary:", file=sys.stderr)
        print(f"  Trajectories: {self.trajectories_processed} processed, {self.trajectories_failed} failed", file=sys.stderr)
        if self.real_bids_found or self.placeholder_bids_generated:
            total_bids = self.real_bids_found + self.placeholder_bids_generated
            print(f"  Bids: {self.real_bids_found}/{total_bids} real ({100*self.real_bids_found/total_bids:.1f}%), {self.placeholder_bids_generated} placeholders", file=sys.stderr)
        if self.first_role_fixes:
            print(f"  First role fixes: {self.first_role_fixes}", file=sys.stderr)
        if self.message_merges:
            print(f"  Message merges: {self.message_merges} (reduced {self.total_messages_before_merge} -> {self.total_messages_after_merge} msgs)", file=sys.stderr)
        if self.finish_appends:
            print(f"  Finish appends: {self.finish_appends}", file=sys.stderr)
        if self.axtrees_truncated:
            print(f"  Axtrees truncated: {self.axtrees_truncated}", file=sys.stderr)
        if self.early_terminations:
            print(f"  Early terminations: {self.early_terminations} trajectories, {self.events_skipped} events skipped", file=sys.stderr)
        if self.truncations:
            print(f"  Truncations: {self.truncations}", file=sys.stderr)
        if self.samples_dropped_over_budget:
            print(f"  Dropped (over budget): {self.samples_dropped_over_budget}", file=sys.stderr)
        if self.bids_referenced_in_actions:
            total = self.bids_referenced_in_actions
            found = self.bids_found_in_axtree
            missing = self.bids_missing_from_axtree
            pct = 100 * found / total if total else 0
            print(f"  BID survival: {found}/{total} ({pct:.1f}%) — {missing} missing", file=sys.stderr)


stats = ConversionStats()


def truncate_axtree_text(axtree: str, max_chars: int = DEFAULT_MAX_AXTREE_CHARS) -> str:
    """Truncate raw axtree text to a character limit, preserving complete lines.

    Drops trailing lines to stay within the limit. Leading lines are kept
    because top-level elements (navigation, headers, interactive elements)
    appear first in axtrees.

    Args:
        axtree: Raw axtree string (before template wrapping).
        max_chars: Maximum character length for the axtree text.

    Returns:
        Truncated axtree string, or original if already within limit.
    """
    if not axtree or len(axtree) <= max_chars:
        return axtree

    # Find last complete line within limit
    truncation_point = axtree.rfind('\n', 0, max_chars)
    if truncation_point <= 0:
        truncation_point = max_chars

    original_lines = axtree.count('\n') + 1
    kept_lines = axtree[:truncation_point].count('\n') + 1
    trimmed_lines = original_lines - kept_lines

    stats.axtrees_truncated += 1
    return axtree[:truncation_point] + f"\n[... {trimmed_lines} lines trimmed ...]"


# Regex for finding axtree content within formatted messages (used by
# _trim_axtree_in_messages to trim oversized messages after truncation).
# Mirrors the markers added by get_web_user_message().
_AXTREE_RE = re.compile(
    r"(============== BEGIN accessibility tree ==============\n)"
    r"(.*?)"
    r"(\n============== END accessibility tree ==============)",
    re.DOTALL,
)


def _trim_axtree_in_messages(conversations: list[dict], overflow_chars: int) -> None:
    """Trim axtree content within formatted messages to reduce total size.

    Targets messages containing accessibility tree markers. Trims trailing
    lines from the axtree content. Handles multiple axtree blocks per message
    (which occur when consecutive WebObservations are merged). Modifies
    conversations in-place.

    Args:
        conversations: Message list to modify in-place.
        overflow_chars: Number of characters to remove.
    """
    # Shared across the re.sub callback via nonlocal
    remaining = overflow_chars

    for msg in conversations:
        if remaining <= 0:
            break

        def _trimmer(match):
            """Callback for re.sub: trim each axtree block found."""
            nonlocal remaining
            if remaining <= 0:
                return match.group(0)

            prefix = match.group(1)
            tree_content = match.group(2)
            suffix = match.group(3)
            notice = "\n[... axtree truncated to fit context window ...]"
            trim_amount = min(remaining + len(notice), len(tree_content))
            keep_chars = max(0, len(tree_content) - trim_amount)

            # Find last complete line boundary
            if keep_chars > 0:
                line_boundary = tree_content.rfind("\n", 0, keep_chars)
                if line_boundary > 0:
                    keep_chars = line_boundary

            trimmed = tree_content[:keep_chars] + notice
            remaining -= len(tree_content) - len(trimmed)
            return prefix + trimmed + suffix

        msg["value"] = _AXTREE_RE.sub(_trimmer, msg["value"])


def format_annotations_as_axtree(
    annotations: list, round_float_sigfigs: int | None = None,
) -> str:
    """Format image annotations in pseudo-axtree format.

    Mirrors the web observation axtree format for consistency:
    ============== BEGIN element annotations ==============
    [1] TEXT 'Oct7' at (0.12, 0.07, 0.08, 0.01)
    [2] ICON_PLAY [clickable] at (0.47, 0.75, 0.03, 0.03)
    ============== END element annotations ==============

    Args:
        annotations: List of annotation objects with element_type, text,
            bounding_box, clickable, and editable attributes.
        round_float_sigfigs: If set, round bounding box coordinates to this
            many significant figures to reduce token waste.
    """
    if not annotations:
        return ""

    def _fmt(v: float) -> str:
        if round_float_sigfigs is not None:
            v = round_sigfigs(v, round_float_sigfigs)
        return str(v)

    lines = ["============== BEGIN element annotations =============="]
    for idx, annotation in enumerate(annotations):
        parts = [f"[{idx + 1}]"]

        if hasattr(annotation, "element_type") and annotation.element_type:
            parts.append(annotation.element_type)

        if hasattr(annotation, "text") and annotation.text:
            parts.append(f"'{annotation.text}'")
        elif hasattr(annotation, "content_description") and annotation.content_description:
            parts.append(f"'{annotation.content_description}'")

        attrs = []
        if hasattr(annotation, "clickable") and annotation.clickable:
            attrs.append("clickable")
        if hasattr(annotation, "editable") and annotation.editable:
            attrs.append("editable")
        if attrs:
            parts.append(f"[{', '.join(attrs)}]")

        if hasattr(annotation, "bounding_box") and annotation.bounding_box:
            bb = annotation.bounding_box
            parts.append(
                f"at ({_fmt(bb.x)}, {_fmt(bb.y)}, {_fmt(bb.width)}, {_fmt(bb.height)})"
            )

        lines.append(" ".join(parts))

    lines.append("============== END element annotations ==============")
    return "\n".join(lines)


def estimate_image_tokens(
    image_path: str,
    media_dir: str | None = None,
    fallback_tokens: int = 1400,
    image_max_pixels: int | None = None,
    image_min_pixels: int | None = None,
) -> int:
    """Estimate the number of vision tokens for an image based on its dimensions.

    Qwen2-VL/Qwen3-VL use 28x28 pixels per token. This reads only the file
    header (no pixel decode), so it's fast.

    Args:
        image_path: Path to the image file (absolute or relative).
        media_dir: Base directory for resolving relative image paths.
        fallback_tokens: Token count to use if the image can't be opened.
        image_max_pixels: If set, clamp pixel count to this maximum before
            computing tokens.
        image_min_pixels: If set, clamp pixel count to this minimum before
            computing tokens.

    Returns:
        Estimated number of vision tokens.
    """
    path = Path(image_path)
    if not path.is_absolute() and media_dir:
        path = Path(media_dir) / path

    try:
        with Image.open(path) as img:
            w, h = img.size
    except Exception:
        return fallback_tokens

    pixels = w * h
    if image_max_pixels is not None:
        pixels = min(pixels, image_max_pixels)
    if image_min_pixels is not None:
        pixels = max(pixels, image_min_pixels)

    return math.ceil(pixels / (28 * 28))


def truncate_to_token_budget(
    conversations: list[dict],
    images: list[str],
    max_tokens: int = 24000,
    system_prompt_tokens: int | None = None,
    media_dir: str | None = None,
    image_max_pixels: int | None = None,
    image_min_pixels: int | None = None,
) -> tuple[list[dict] | None, list[str]]:
    """Truncate conversation turns to fit within total token budget.

    Counts text tokens using digit-aware estimation (see estimate_text_tokens)
    and image tokens based on actual image dimensions when available (falls
    back to a conservative estimate). This prevents LLaMA Factory's cutoff_len
    truncation from removing image tokens while leaving the images list intact
    (feature/token mismatch).

    Returns (None, []) if the sample should be dropped because it exceeds
    the budget even after axtree trimming.

    Args:
        conversations: List of message dicts with "from" and "value" keys.
        images: Ordered list of image paths corresponding to <image> tags.
        max_tokens: Maximum total token budget.
        system_prompt_tokens: Tokens reserved for the system prompt. Defaults
            to the cached estimate from the rendered system prompt.
        media_dir: Base directory for resolving relative image paths.
        image_max_pixels: If set, clamp image pixel count to this max.
        image_min_pixels: If set, clamp image pixel count to this min.
    """
    if system_prompt_tokens is None:
        system_prompt_tokens = _SYSTEM_PROMPT_TOKENS

    def count_images(msg: dict) -> int:
        return msg.get("value", "").count("<image>")

    # Pre-compute per-image token costs using actual dimensions
    image_token_costs = [
        estimate_image_tokens(
            img, media_dir=media_dir,
            image_max_pixels=image_max_pixels,
            image_min_pixels=image_min_pixels,
        )
        for img in images
    ]

    available = max_tokens - system_prompt_tokens
    total_tokens = 0
    keep_count = 0
    image_idx = 0  # tracks position in the images list

    for msg in conversations:
        text_tokens = estimate_text_tokens(msg.get("value", ""))
        n_images = count_images(msg)

        # Sum actual token costs for this message's images
        image_tokens = sum(
            image_token_costs[image_idx + i]
            for i in range(n_images)
            if image_idx + i < len(image_token_costs)
        )

        msg_tokens = text_tokens + image_tokens

        if total_tokens + msg_tokens <= available:
            total_tokens += msg_tokens
            keep_count += 1
            image_idx += n_images
        else:
            break

    # Ensure we keep at least 2 messages (one human, one gpt) and even count
    if keep_count < 2:
        keep_count = min(2, len(conversations))
    if keep_count % 2 == 1:
        keep_count = max(2, keep_count - 1)

    # Truncate message count if needed
    if keep_count < len(conversations):
        stats.truncations += 1
        conversations = conversations[:keep_count]
        kept_image_count = sum(count_images(m) for m in conversations)
        images = images[:kept_image_count]

    # Check if kept messages exceed budget (can happen even without message-count
    # truncation, e.g. a 2-message conversation where the first message is enormous)
    kept_image_count = sum(count_images(m) for m in conversations)
    kept_text_tokens = sum(
        estimate_text_tokens(m.get("value", "")) for m in conversations
    )
    kept_image_tokens = sum(image_token_costs[:kept_image_count])
    kept_total = kept_text_tokens + kept_image_tokens

    if kept_total > available:
        overflow_chars = (kept_total - available) * 3
        _trim_axtree_in_messages(conversations, overflow_chars)

        # Final check: if still over budget (e.g. non-axtree text is enormous), drop
        final_text_tokens = sum(
            estimate_text_tokens(m.get("value", "")) for m in conversations
        )
        if final_text_tokens + kept_image_tokens > available:
            stats.samples_dropped_over_budget += 1
            return None, []

    return conversations, images


def verify_args(required_args, optional_args, input_args):
    # all required args should be included
    for arg in required_args:
        if arg not in input_args:
            return False
    # all input args should be one of the specified args of the function
    for arg in input_args:
        if arg not in required_args + optional_args:
            return False
    return True


# Convert function call to OH format
def format_function(function_name, parameters):
    # Example OH function format:
    """
    <function=example_function_name>
    <parameter=example_parameter_1>value_1</parameter>
    <parameter=example_parameter_2>
    This is the value for the second parameter
    that can span
    multiple lines
    </parameter>
    </function>
    """
    function_call = ""
    for parameter in parameters:
        value = parameters[parameter]
        function_call += f"<parameter={parameter}>\n{value}\n</parameter>\n"
    function_call = f"<function={function_name}>\n{function_call}</function>"
    return function_call


# Extract the tool in a OH format function call
def extract_function_call(content):
    for tool in openhands_default_tools:
        if f"<function={tool}" in content:
            return tool
    return None


PREV_BID = None

# BIDs present in the most recently generated axtree, used for survival tracking.
# Reset at the start of each trajectory in process_row().
_CURRENT_AXTREE_BIDS: set[str] = set()

# Regex to extract BIDs from axtree text. Matches [bid] at start of line or after
# indentation (tab characters), which is the BrowserGym axtree format.
_BID_EXTRACT_RE = re.compile(r"^\t*\[(\w+)\] ", re.MULTILINE)


def _extract_bids_from_axtree(axtree: str) -> set[str]:
    """Extract all BIDs from an axtree string."""
    return set(_BID_EXTRACT_RE.findall(axtree))


def standardized_event_to_openhands_message(
    id,
    event: ApiAction | CodeAction | MessageAction | TextObservation | WebObservation,
    previous_web_actions: list,
    is_web: bool,
    api_env: str = None,
    api_sigs=None,
    languages: list | None = None,
    max_axtree_chars: int = DEFAULT_MAX_AXTREE_CHARS,
    round_float_sigfigs: int | None = None,
    axtree_filter_som: bool = False,
    axtree_filter_visible: bool = False,
) -> dict:
    global PREV_BID, _CURRENT_AXTREE_BIDS
    if languages is None:
        languages = []
    if isinstance(event, WebObservation):
        if event.axtree is not None:
            axtree = event.axtree  # Pre-existing text: no browser needed
        elif event.html is not None:
            # Lazy-init browser only when we actually need to generate from HTML
            gen = get_axtree_generator(
                filter_som=axtree_filter_som, filter_visible=axtree_filter_visible,
            )
            if gen.last_html != event.html:
                axtree = gen.build_axtree(id, event.html, "all")
            else:
                axtree = gen.last_xtree
        else:
            axtree = None  # No HTML or axtree available

        # Strip RSS/XML <image> tags from axtree to prevent false image token counts.
        # RSS 2.0 uses <image> for channel logos; these leak into axtree when browsers
        # render raw RSS feeds (known to affect 22/7882 go-browse-wa trajectories).
        if axtree:
            axtree = axtree.replace("<image>", "").replace("</image>", "")
            axtree = truncate_axtree_text(axtree, max_axtree_chars)
            _CURRENT_AXTREE_BIDS = _extract_bids_from_axtree(axtree)
        else:
            _CURRENT_AXTREE_BIDS = set()

        prompt = get_web_user_message("", event.url, axtree, PREV_BID)

        # Handle nested image observation
        image_path = None
        if hasattr(event, "image_observation") and event.image_observation:
            image_path = event.image_observation.content

            # Add visual observation section (OpenHands format)
            prompt += (
                "\n<image>\n"
                "Image: Current webpage screenshot (Note that only visible portion of "
                "webpage is present in the screenshot. However, the Accessibility tree "
                "contains information from the entire webpage.)\n"
            )

            # Add image annotations if present (for bounding box info not in axtree)
            img_obs = event.image_observation
            if hasattr(img_obs, "annotations") and img_obs.annotations:
                prompt += format_annotations_as_axtree(
                    img_obs.annotations, round_float_sigfigs=round_float_sigfigs,
                ) + "\n"

        return {"from": "human", "value": prompt, "_image_path": image_path}

    if isinstance(event, ApiAction):
        PREV_BID = None
        thought = event.description + "\n\n" if event.description else ""
        function_name = event.function
        arguments = {k: v for k, v in event.kwargs.items() if k not in ["element_id", "xpath"]}
        if round_float_sigfigs is not None:
            arguments = {k: round_sigfigs(v, round_float_sigfigs) for k, v in arguments.items()}

        # for tool that are one of the default OH tools
        if function_name in openhands_default_tools and function_name not in api_sigs:
            tool_args = openhands_default_tools[function_name]
            if not verify_args(tool_args["required"], tool_args["optional"], arguments):
                raise ValueError(f"Function call with wrong argument: {event}")
            function_call = format_function(function_name, arguments)
            return {"from": "function_call", "value": f"{thought}{function_call}"}

        # for OH default browser based apis that don't require bid
        if (
            is_web
            and function_name in browser_default_apis
            and function_name not in api_sigs
            and "bid" not in browser_default_apis[function_name]["required"]
        ):
            api_args = browser_default_apis[function_name]
            if not verify_args(api_args["required"], api_args["optional"], arguments):
                raise ValueError(f"Function call with wrong argument: {event}")
            api_action = f"{function_name}({', '.join([f'{k}={arguments[k]}' for k in arguments])})"
            previous_web_actions.extend([api_action])
            function_call = format_function("browser", {"code": api_action})
            return {"from": "function_call", "value": f"{thought}{function_call}"}

        # try to directly get the browsergym_id from the event kwargs
        browsergym_id = event.kwargs.get("bid", None)
        if not browsergym_id:
            browsergym_id = event.kwargs.get("element_id", None)
        if browsergym_id:
            stats.real_bids_found += 1

        # this gets the browsergym_id of the element that the user is interacting with
        # the latest(last seen) html's obs is updated whenever build_axtree is called
        # the latest obs is used to get the browsergym_id
        if not browsergym_id:
            event_xpath = event.kwargs.get("xpath", None)
            if event_xpath:
                browsergym_id = get_axtree_generator(
                    filter_som=axtree_filter_som, filter_visible=axtree_filter_visible,
                ).get_bid(id, event_xpath, "all")
                if browsergym_id:
                    stats.real_bids_found += 1

        # Generate placeholder bid for web datasets when get_bid fails
        if not browsergym_id and is_web:
            event_xpath = event.kwargs.get("xpath", None)
            if event_xpath:
                # Use xpath hash as placeholder to maintain some consistency
                placeholder_id = f"placeholder_bid_{abs(hash(event_xpath)) % 10000}"
                browsergym_id = f'"{placeholder_id}"'
                stats.placeholder_bids_generated += 1

        # Track BID survival: check if the resolved BID exists in the current axtree.
        # Skip placeholder BIDs (generated when xpath lookup fails) since they're
        # never in the axtree by definition.
        if browsergym_id and _CURRENT_AXTREE_BIDS:
            bid_clean = browsergym_id.strip("'\"")
            if "placeholder_bid_" not in bid_clean:
                stats.bids_referenced_in_actions += 1
                if bid_clean in _CURRENT_AXTREE_BIDS:
                    stats.bids_found_in_axtree += 1
                else:
                    stats.bids_missing_from_axtree += 1

        # for tool calls that are not browser based since there is no browsergym_id
        # and tool calls that are specified as non-web
        # these should all be dataset specific apis
        if not is_web and function_name in api_sigs:
            if not api_env:
                # Default to 'execute_ipython_cell' if api_env is not specified
                api_env = "execute_ipython_cell"
            if not verify_args(
                api_sigs[function_name]["required"], api_sigs[function_name]["optional"], arguments
            ):
                raise ValueError(f"Function call with wrong argument: {event}")
            api_action = f"{function_name}({', '.join([f'{k}={arguments[k]}' for k in arguments])})"
            function_call = format_function(
                api_env, {function_args.get(api_env, "code"): api_action}
            )
            return {"from": "function_call", "value": f"{thought}{function_call}"}

        api_env = "browser"

        if browsergym_id and not browsergym_id[0] == browsergym_id[-1] == '"':
            browsergym_id = f'"{browsergym_id}"'
        PREV_BID = browsergym_id
        # for apis that are browser based but are not OH default browser apis
        # these should all be dataset specific apis
        if function_name in api_sigs:
            if "bid" in api_sigs[function_name]["required"] and browsergym_id:
                arguments["bid"] = browsergym_id
            if not verify_args(
                api_sigs[function_name]["required"], api_sigs[function_name]["optional"], arguments
            ):
                raise ValueError(f"Function call with wrong argument: {event}")
            api_action = f"{function_name}({', '.join([f'{k}={arguments[k]}' for k in arguments])})"
            function_call = format_function(
                api_env, {function_args.get(api_env, "code"): api_action}
            )
            return {"from": "function_call", "value": f"{thought}{function_call}"}

        # for tool calls that are browser based and need bid
        api_args = browser_default_apis[function_name]
        if browsergym_id:
            arguments["bid"] = browsergym_id

        # to handle mismatching "bid" and "id" arguments
        if "bid" not in arguments:
            if "id" in arguments:
                arguments["bid"] = arguments.pop("id")
                PREV_BID = arguments["bid"]
        if not verify_args(api_args["required"], api_args["optional"], arguments):
            raise ValueError(f"Function call with wrong argument: {event}")
        api_action = f"{function_name}({', '.join([f'{k}={arguments[k]}' for k in arguments])})"
        previous_web_actions.extend([api_action])
        function_call = format_function(api_env, {function_args.get(api_env, "code"): api_action})
        return {"from": "function_call", "value": f"{thought}{function_call}"}

    if isinstance(event, CodeAction):
        thought = event.description + "\n\n" if event.description else ""
        function_name = action_function.get(event.language, f"execute_{event.language}")
        code_content = event.content
        if function_name not in openhands_default_tools:
            languages.append(event.language)
            function_name = "execute_ipython_cell"
            code_content = f'{event.language}("{code_content}")'
        arg = function_args.get(function_name, "code")
        code_action = format_function(function_name, {arg: code_content})
        return {"from": "function_call", "value": f"{thought}{code_action}"}

    elif isinstance(event, MessageAction):
        thought = event.description + "\n\n" if event.description else ""
        if "<finish>" in event.content and "</finish>" in event.content:
            match = re.search(r"<finish>(.*?)</finish>", event.content, re.DOTALL)
            content = match.group(1).strip()
            finish_function_call = format_function(
                "finish", {"message": content, "task_completed": "true"}
            )
            return {"from": "function_call", "value": f"{thought}{finish_function_call}"}
        return {"from": "gpt", "value": f"{thought}{event.content}"}

    elif isinstance(event, TextObservation):
        if event.source == "user":
            event.source = "human"

        elif event.source == "agent":
            event.source = "gpt"

        elif event.source == "environment":
            event.source = "observation"

        else:
            raise ValueError(f"Wrong event source: {event.source}")
        return {"from": event.source, "value": event.content}

    elif isinstance(event, ImageObservation):
        # Handle ImageObservation with pseudo-axtree format for annotations
        annotation_tree = ""
        if hasattr(event, "annotations") and event.annotations:
            annotation_tree = format_annotations_as_axtree(
                event.annotations, round_float_sigfigs=round_float_sigfigs,
            )

        # Build the observation value
        value = "<image>\nImage: Current screen observation.\n"
        if annotation_tree:
            value += annotation_tree

        return {
            "from": "observation",
            "value": value,
            "_image_path": event.content,
        }

    else:
        raise ValueError(f"Unknown event type: {type(event)}\n{event}")


def process_row(line, is_web, api_env, api_tool_description, api_sigs,
                export_for="explicit", media_dir=None,
                max_axtree_chars=DEFAULT_MAX_AXTREE_CHARS, max_tokens=24000,
                image_max_pixels=None, image_min_pixels=None,
                round_float_sigfigs=None,
                axtree_filter_som=False, axtree_filter_visible=False):
    global _CURRENT_AXTREE_BIDS, PREV_BID
    _CURRENT_AXTREE_BIDS = set()
    PREV_BID = None

    std_data = json.loads(line)
    trajectory = Trajectory(**std_data)
    id = trajectory.id
    events = trajectory.content
    # details = trajectory.details
    conversations = []
    previous_web_actions = []
    languages = []
    image_paths = []

    # Budget tracking for early termination of expensive operations
    available_chars = (max_tokens - _SYSTEM_PROMPT_TOKENS) * 3

    for i in range(len(events)):
        event = events[i]

        # Early termination: skip remaining events once budget is clearly exceeded.
        # Post-processing (merging, truncation) will handle the conversation state.
        running_chars = sum(len(m.get("value", "")) for m in conversations)
        if running_chars > available_chars and len(conversations) >= 2:
            stats.early_terminations += 1
            stats.events_skipped += len(events) - i
            break

        try:
            message = standardized_event_to_openhands_message(
                id, event, previous_web_actions, is_web, api_env, api_sigs, languages,
                max_axtree_chars=max_axtree_chars,
                round_float_sigfigs=round_float_sigfigs,
                axtree_filter_som=axtree_filter_som,
                axtree_filter_visible=axtree_filter_visible,
            )
            if not message:
                return None

            # Extract image path if present
            if "_image_path" in message:
                path = message.pop("_image_path")
                if path:
                    image_paths.append(path)

            if len(conversations) == 0:
                # append api function docs to first user message when available
                if api_env:
                    message["value"] = api_tool_description + message["value"]
                conversations.extend([message])
                continue

            # Combine consecutive user message and web observation
            if conversations[-1]["from"] == "human" and isinstance(event, WebObservation):
                conversations[-1]["value"] += "\n\n" + message["value"]
                continue

            # Match observations to function_calls
            if conversations[-1]["from"] == "function_call" and isinstance(event, TextObservation):
                message["from"] = "observation"
                function_name = extract_function_call(conversations[-1]["value"])
                if function_name:
                    message["value"] = (
                        f"EXECUTION RESULT of [{function_name}]:\n" + message["value"]
                    )

            conversations.extend([message])

        except Exception as e:
            traceback.print_exc()
            print(e, file=sys.stderr)
            return None
    if languages:
        language_descriptions = get_language_descriptions(languages)
        conversations[0]["value"] = language_descriptions + "\n\n" + conversations[0]["value"]
    for m in conversations:
        if export_for == "training" and m["from"] == "function_call":
            m["from"] = "gpt"
        if m["from"] == "observation":
            m["from"] = "human"

    # --- Post-processing for valid ShareGPT format ---
    # 1. Fix first message role (must be "human" for ShareGPT)
    if conversations and conversations[0]["from"] != "human":
        stats.first_role_fixes += 1
        conversations[0]["from"] = "human"

    # 2. Merge consecutive same-role messages
    pre_merge_count = len(conversations)
    if len(conversations) > 1:
        merged = [conversations[0]]
        for msg in conversations[1:]:
            if msg["from"] == merged[-1]["from"]:
                merged[-1]["value"] += "\n\n" + msg["value"]
            else:
                merged.append(msg)
        conversations = merged
    if len(conversations) != pre_merge_count:
        stats.message_merges += 1
        stats.total_messages_before_merge += pre_merge_count
        stats.total_messages_after_merge += len(conversations)

    # 3. Ensure even message count (required for prompt/response pairing)
    #    After steps 1+2, sequence strictly alternates starting with "human",
    #    so odd count means trailing "human" (observation). Append finish response.
    if len(conversations) % 2 == 1:
        conversations.append({"from": "gpt", "value": "<finish>Task completed.</finish>"})
        stats.finish_appends += 1

    # 4. Truncate to fit token budget (prevents LLaMA Factory truncation mismatch)
    conversations, image_paths = truncate_to_token_budget(
        conversations, image_paths, max_tokens=max_tokens, media_dir=media_dir,
        image_max_pixels=image_max_pixels, image_min_pixels=image_min_pixels,
    )

    # Sample dropped: still over budget even after axtree trimming
    if conversations is None:
        return None

    # 5. Validate <image> token count matches images list
    token_count = sum(m.get("value", "").count("<image>") for m in conversations)
    if token_count != len(image_paths):
        print(
            f"WARNING: {trajectory.id}: {token_count} <image> tokens "
            f"but {len(image_paths)} images",
            file=sys.stderr,
        )

    output = {
        "id": trajectory.id,
        "conversations": conversations,
        "system": _SYSTEM_PROMPT,
    }

    if image_paths:
        output["images"] = image_paths

    return output


def process_line(line, is_web, api_env, export_for="explicit", media_dir=None,
                 max_axtree_chars=DEFAULT_MAX_AXTREE_CHARS, max_tokens=24000,
                 image_max_pixels=None, image_min_pixels=None,
                 round_float_sigfigs=None,
                 axtree_filter_som=False, axtree_filter_visible=False):
    exclude_apis = browser_default_apis if is_web else {}
    api_tool_description, api_sigs = get_api_tool_description(dataset, exclude_apis, api_env)
    output_line = process_row(
        line,
        is_web=is_web,
        api_env=api_env,
        api_tool_description=api_tool_description,
        api_sigs=api_sigs,
        export_for=export_for,
        media_dir=media_dir,
        max_axtree_chars=max_axtree_chars,
        max_tokens=max_tokens,
        image_max_pixels=image_max_pixels,
        image_min_pixels=image_min_pixels,
        round_float_sigfigs=round_float_sigfigs,
        axtree_filter_som=axtree_filter_som,
        axtree_filter_visible=axtree_filter_visible,
    )
    if output_line is None:
        return None
    output_line = json.dumps(output_line)
    return output_line


def main():
    parser = argparse.ArgumentParser(description="Convert standardized data to SFT format")
    parser.add_argument(
        "--is_web",
        type=str,
        choices=["yes", "no"],
        help="Does the dataset contain web api",
        required=True,
        default="no",
    )
    parser.add_argument(
        "--api_env",
        type=str,
        choices=list(openhands_default_tools.keys()) + [None],
        help="The environment in which the APIs are pre-defined",
        default=None,
    )
    parser.add_argument(
        "--export_for",
        type=str,
        choices=["explicit", "training"],
        default="explicit",
        help="'explicit' preserves function_call message role, 'training' replaces it with gpt role for LLaMA Factory",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input file path (default: stdin)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip already processed IDs)",
    )
    parser.add_argument(
        "--media-dir",
        type=str,
        default=None,
        help="Base directory for resolving relative image paths (enables accurate image token estimation)",
    )
    parser.add_argument(
        "--max-axtree-chars",
        type=int,
        default=DEFAULT_MAX_AXTREE_CHARS,
        help=f"Max characters per individual axtree (default: {DEFAULT_MAX_AXTREE_CHARS}, "
             f"~{DEFAULT_MAX_AXTREE_CHARS // 3}k tokens). Axtrees exceeding this are "
             "truncated by dropping trailing lines.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=24000,
        help="Max total token budget per example (text + images). Default: 24000",
    )
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=1048576,
        help="Max pixels per image for token estimation (matches training config). Default: 1048576",
    )
    parser.add_argument(
        "--image-min-pixels",
        type=int,
        default=4096,
        help="Min pixels per image for token estimation (matches training config). Default: 4096",
    )
    parser.add_argument(
        "--round-float-sigfigs",
        type=int,
        default=None,
        help="Round float values (bounding box coordinates, action parameters) to N "
             "significant figures to reduce token waste from high-precision floats. "
             "Default: None (no rounding).",
    )
    parser.add_argument(
        "--axtree-filter-som",
        action="store_true",
        default=False,
        help="Filter axtree to set-of-marks (interactive) elements only. "
             "Only affects datasets where axtree is generated from HTML "
             "(e.g. weblinx, wonderbread, webarena). Pre-existing axtree text "
             "is not filtered. Reduces token usage by ~60-80%% for web datasets.",
    )
    parser.add_argument(
        "--axtree-filter-visible",
        action="store_true",
        default=False,
        help="Filter axtree to visible elements only (visibility >= 0.5). "
             "Removes hidden menus, collapsed sections, off-screen elements. "
             "Only affects datasets where axtree is generated from HTML.",
    )
    args = parser.parse_args()
    args.is_web = args.is_web == "yes"

    # Load processed IDs if resuming
    processed_ids = set()
    if args.resume and args.output and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data and "id" in data:
                        processed_ids.add(data["id"])
                except json.JSONDecodeError:
                    continue
        print(f"Resuming: {len(processed_ids)} already processed", file=sys.stderr)

    # Count total for progress bar (if input file)
    total = None
    if args.input:
        with open(args.input) as f:
            total = sum(1 for _ in f)

    # Set up input/output sources
    input_source = open(args.input) if args.input else sys.stdin
    output_dest = open(args.output, "a" if (args.resume and processed_ids) else "w") if args.output else sys.stdout

    stats.reset()
    try:
        for line in tqdm(input_source, total=total, desc="SFT conversion", file=sys.stderr, mininterval=30):
            # Parse to get ID for skip check
            std_data = json.loads(line)
            traj_id = std_data.get("id")
            if traj_id in processed_ids:
                continue

            result = process_line(
                line, args.is_web, args.api_env, args.export_for, args.media_dir,
                args.max_axtree_chars, args.max_tokens,
                args.image_max_pixels, args.image_min_pixels,
                args.round_float_sigfigs,
                args.axtree_filter_som, args.axtree_filter_visible,
            )
            if result is None:
                stats.trajectories_failed += 1
                continue
            stats.trajectories_processed += 1
            output_dest.write(result + "\n")
            if args.output:
                output_dest.flush()  # Immediate write for resume safety
    finally:
        if args.input:
            input_source.close()
        if args.output:
            output_dest.close()
        stats.print_summary()


if __name__ == "__main__":
    main()
