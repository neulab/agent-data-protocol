import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import browsergym.core  # Register browsergym environments with gymnasium
import gymnasium as gym
import requests
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str
from lxml import etree


# Module-level cache for host check results (True = responds, False = timeout)
_host_cache = {}


def check_host_responds(host, timeout=2):
    """Check if a host responds within timeout. Returns False only for timeouts."""
    # Check cache first
    if host in _host_cache:
        return _host_cache[host]

    try:
        # Try HTTP connection to the host
        url = f"http://{host}/"
        requests.head(url, timeout=(timeout, timeout), allow_redirects=True)
        _host_cache[host] = True
        return True
    except requests.exceptions.Timeout:
        _host_cache[host] = False
        return False  # Timeout = dead/slow host
    except requests.exceptions.ConnectionError:
        _host_cache[host] = True
        return True   # Connection refused = fast failure, browser handles it
    except Exception:
        _host_cache[host] = True
        return True   # Other errors = fast failure


def neutralize_slow_hosts(html, timeout=2, max_total_time=15):
    """Replace URLs pointing to timeout hosts with placeholders.

    This prevents the browser from hanging on dead servers when loading
    any resource type (scripts, CSS, images, iframes, etc.).

    Args:
        html: HTML content
        timeout: Per-host timeout in seconds
        max_total_time: Maximum total time for all host checks

    Returns:
        Modified HTML with slow host URLs replaced by about:blank
    """
    from urllib.parse import urlparse

    # Pattern matches resource-loading attributes with URLs
    # Captures: (1) attribute prefix like 'src="', (2) the URL, (3) closing quote
    url_pattern = r'((?:src|href|data|poster|action)=["\'])(https?://[^"\']+)(["\'])'

    # Extract unique hosts from URLs
    all_hosts = set()
    for m in re.finditer(url_pattern, html, re.IGNORECASE):
        url = m.group(2)
        try:
            parsed = urlparse(url)
            if parsed.netloc:
                all_hosts.add(parsed.netloc)
        except Exception:
            pass

    if not all_hosts:
        return html

    # Separate cached vs uncached hosts
    slow_hosts = set()
    hosts_to_check = set()
    for host in all_hosts:
        if host in _host_cache:
            if not _host_cache[host]:  # Cached as slow
                slow_hosts.add(host)
        else:
            hosts_to_check.add(host)

    # Check uncached hosts in parallel
    if hosts_to_check:
        with ThreadPoolExecutor(max_workers=min(10, len(hosts_to_check))) as executor:
            futures = {host: executor.submit(check_host_responds, host, timeout)
                       for host in hosts_to_check}

            for host, future in futures.items():
                try:
                    if not future.result(timeout=max_total_time):
                        slow_hosts.add(host)
                except Exception:
                    # Timeout or error getting result - assume slow
                    slow_hosts.add(host)
                    _host_cache[host] = False

    if not slow_hosts:
        return html

    # Replace URLs pointing to slow hosts with about:blank
    def replacer(m):
        prefix, url, suffix = m.group(1), m.group(2), m.group(3)
        try:
            parsed = urlparse(url)
            if parsed.netloc in slow_hosts:
                return prefix + "about:blank" + suffix
        except Exception:
            pass
        return m.group(0)

    return re.sub(url_pattern, replacer, html, flags=re.IGNORECASE)


class HTMLToAXTree:
    def __init__(self, dataset: str, timeout_ms: int = 30000,
                 filter_som_only: bool = False, filter_visible_only: bool = False):
        """Initialize HTML to accessibility tree converter.

        Args:
            dataset: Dataset name, used for error log filenames.
            timeout_ms: Playwright timeout in milliseconds for all browser
                operations. Prevents indefinite hangs on pages with
                unresolvable resources. Defaults to 30 seconds.
            filter_som_only: If True, only include set-of-marks (interactive)
                elements in the axtree. Requires extra_element_properties from
                BrowserGym observation.
            filter_visible_only: If True, only include visible elements
                (visibility >= 0.5) in the axtree.
        """
        self.errors = []
        self.dataset = dataset
        self.filter_som_only = filter_som_only
        self.filter_visible_only = filter_visible_only
        self.env = gym.make(
            "browsergym/openended",
            headless=True,
            task_kwargs={"start_url": "about:blank"},
            wait_for_user_message=False,
            tags_to_mark="all",
            timeout=timeout_ms,
        )
        self._initialized = False
        self.last_html = None
        self.last_xtree = None
        self.last_obs = None

    def build_axtree(self, id, html_content: str, chunk) -> str:
        self.last_html = html_content
        temp_dir = os.path.abspath("./temp/")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f"temp_{self.dataset}_{id}.html")

        # Replace URLs pointing to dead/slow hosts with placeholders
        html_content = neutralize_slow_hosts(html_content)

        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Only reset once at first use, then reuse browser for all subsequent pages
        if not self._initialized:
            self.env.reset()
            self._initialized = True

        # Navigate to the temp file.
        # Timeout is set at the Playwright context level via gym.make(timeout=...).
        # If the page hangs (e.g. unresolvable resources despite neutralize_slow_hosts),
        # the step will raise rather than block indefinitely.
        try:
            obs, reward, terminated, truncated, info = self.env.step(
                f"goto('file://{temp_file}')"
            )
        except Exception as e:
            print(f"Warning: build_axtree failed for {id}: {e}", file=sys.stderr)
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return ""

        if os.path.exists(temp_file):
            os.remove(temp_file)

        self.last_obs = obs

        # When filtering is active, pass extra_element_properties so
        # flatten_axtree_to_str can filter by SoM / visibility flags.
        # Fall back to unfiltered if properties are unexpectedly missing.
        needs_extra = self.filter_som_only or self.filter_visible_only
        extra_props = obs.get("extra_element_properties") if needs_extra else None
        if needs_extra and extra_props is None:
            print(f"Warning: axtree filter requested but extra_element_properties "
                  f"missing for {id}", file=sys.stderr)
            som_flag = False
            vis_flag = False
        else:
            som_flag = self.filter_som_only
            vis_flag = self.filter_visible_only

        self.last_xtree = flatten_axtree_to_str(
            obs["axtree_object"],
            extra_properties=extra_props,
            filter_som_only=som_flag,
            filter_visible_only=vis_flag,
        )

        return self.last_xtree

    def get_bid(self, id, x_path: str, chunk) -> str:
        if self.last_obs is None:
            print(f"Warning: get_bid called but last_obs is None (id={id}, xpath={x_path})", file=sys.stderr)
            return None
        html_string = flatten_dom_to_str(self.last_obs["dom_object"])
        tree = etree.HTML(html_string)
        try:
            if len(x_path) >= 2 and x_path[0] == x_path[-1] and x_path[0] in ('"', "'"):
                x_path = x_path[1:-1]
            element = tree.xpath(x_path)
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
    html_to_axtree = HTMLToAXTree("test")
    print(html_to_axtree.build_axtree("test_id", "<html><body><h1>Hello World</h1></body></html>", "chunk1"))
    html_to_axtree.env.close()
