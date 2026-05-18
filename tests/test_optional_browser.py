"""Regression test: the OpenHands SFT pipeline must import cleanly without browsergym.

`agents/openhands/system_prompt/tools/__init__.py` wraps the `BrowserTool` import in
`try/except ModuleNotFoundError` so that converters for non-web datasets work on
environments that have not installed `browsergym`. This test simulates that
condition by injecting a meta-path finder that raises ModuleNotFoundError for any
`browsergym*` import, then reloads the affected modules.
"""

import importlib
import sys


class _BlockBrowserGym:
    """Meta-path finder that pretends browsergym is uninstalled."""

    def find_module(self, name, path=None):  # py<3.4-compatible API still honored
        if name.startswith("browsergym"):
            return self
        return None

    def load_module(self, name):  # pragma: no cover - exercised via import machinery
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)


def _reload_target_modules():
    for name in list(sys.modules):
        if (
            name.startswith("browsergym")
            or name.startswith("agents.openhands.system_prompt.tools")
            or name == "agents.openhands.system_prompt.system"
        ):
            del sys.modules[name]


def test_tools_init_handles_missing_browsergym():
    finder = _BlockBrowserGym()
    sys.meta_path.insert(0, finder)
    try:
        _reload_target_modules()
        tools = importlib.import_module("agents.openhands.system_prompt.tools")
        assert tools.BrowserTool is None
    finally:
        sys.meta_path.remove(finder)
        _reload_target_modules()


def test_get_system_message_without_browsing_omits_browser_tools():
    finder = _BlockBrowserGym()
    sys.meta_path.insert(0, finder)
    try:
        _reload_target_modules()
        system = importlib.import_module("agents.openhands.system_prompt.system")
        message = system.get_system_message(codeact_enable_browsing=False)
        assert "BrowserTool" not in message
    finally:
        sys.meta_path.remove(finder)
        _reload_target_modules()
