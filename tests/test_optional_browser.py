"""Regression test for #213: OpenHands tool imports must tolerate a missing browsergym.

`agents/openhands/system_prompt/tools/__init__.py` wraps the `BrowserTool`
re-export in `try/except ModuleNotFoundError` so that converters for
non-web datasets work on environments that have not installed
`browsergym`. We exercise that path by running a subprocess with a
`sys.meta_path` finder that raises `ModuleNotFoundError` for every
`browsergym*` import.

The subprocess is used (rather than monkeypatching the in-process
import system) because the tool modules import third-party packages
such as `litellm` at module load time, and those packages may or may
not be installed in the test environment; reloading them through an
in-process patch is brittle. The subprocess inherits the parent
interpreter and PYTHONPATH so the test still runs against the actual
repository code, but it gets a clean import graph each time.
"""

import os
import subprocess
import sys
import textwrap

import pytest


def _run_block_browsergym_subprocess(script: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    env["MY_DATASET"] = "optional_browser_test"
    return subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


_BLOCK_BROWSERGYM_PREAMBLE = textwrap.dedent(
    """
    import sys


    class _BlockBrowserGym:
        def find_module(self, name, path=None):
            if name.startswith("browsergym"):
                return self
            return None

        def load_module(self, name):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)


    sys.meta_path.insert(0, _BlockBrowserGym())
    """
)


def test_tools_init_handles_missing_browsergym():
    pytest.importorskip("litellm")
    result = _run_block_browsergym_subprocess(
        _BLOCK_BROWSERGYM_PREAMBLE
        + textwrap.dedent(
            """
            from agents.openhands.system_prompt import tools

            assert tools.BrowserTool is None, (
                f"BrowserTool should be None when browsergym is unavailable, "
                f"got {tools.BrowserTool!r}"
            )
            print("OK")
            """
        )
    )
    assert result.returncode == 0, (
        f"subprocess failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_get_system_message_without_browsing_omits_browser_tools():
    pytest.importorskip("litellm")
    result = _run_block_browsergym_subprocess(
        _BLOCK_BROWSERGYM_PREAMBLE
        + textwrap.dedent(
            """
            from agents.openhands.system_prompt.system import get_system_message

            message = get_system_message(codeact_enable_browsing=False)
            assert "BrowserTool" not in message, (
                "get_system_message(codeact_enable_browsing=False) must not "
                "advertise BrowserTool"
            )
            print("OK")
            """
        )
    )
    assert result.returncode == 0, (
        f"subprocess failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout
