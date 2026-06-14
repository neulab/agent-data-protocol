import importlib
import json
import sys
import types


def import_openhands_v0_converter(monkeypatch):
    """Import std_to_sft without constructing the browsergym environment."""
    monkeypatch.setenv("MY_DATASET", "role_preservation_test")

    fake_html_module = types.ModuleType("scripts.html_to_axtree")

    class FakeHTMLToAXTree:
        def __init__(self, dataset):
            self.dataset = dataset
            self.last_html = None
            self.last_xtree = None

        def build_axtree(self, id, html_content, chunk):
            self.last_html = html_content
            self.last_xtree = ""
            return ""

        def get_bid(self, id, x_path, chunk):
            return None

    fake_html_module.HTMLToAXTree = FakeHTMLToAXTree
    monkeypatch.setitem(sys.modules, "scripts.html_to_axtree", fake_html_module)

    fake_system_module = types.ModuleType("agents.openhands_v0.system_prompt.system")
    fake_system_module.get_system_message = lambda *args, **kwargs: "system prompt"
    monkeypatch.setitem(sys.modules, "agents.openhands_v0.system_prompt.system", fake_system_module)

    fake_user_module = types.ModuleType("agents.openhands_v0.system_prompt.user")
    fake_user_module.get_web_user_message = lambda *args, **kwargs: "web prompt"
    monkeypatch.setitem(sys.modules, "agents.openhands_v0.system_prompt.user", fake_user_module)

    monkeypatch.delitem(sys.modules, "agents.openhands_v0.std_to_sft", raising=False)
    return importlib.import_module("agents.openhands_v0.std_to_sft")


def atif_line(trajectory_id, steps):
    return json.dumps(
        {
            "schema_version": "ATIF-v1.7",
            "trajectory_id": trajectory_id,
            "agent": {"name": "role_preservation_test", "version": "test"},
            "steps": steps,
        }
    )


def test_openhands_v0_converter_preserves_tool_roles(monkeypatch):
    converter = import_openhands_v0_converter(monkeypatch)
    line = atif_line(
        "role-preservation",
        [
            {"step_id": 1, "source": "user", "message": "Run pwd."},
            {
                "step_id": 2,
                "source": "agent",
                "message": "",
                "tool_calls": [
                    {
                        "tool_call_id": "call_000001",
                        "function_name": "execute_bash",
                        "arguments": {"command": "pwd"},
                    }
                ],
                "observation": {
                    "results": [{"source_call_id": "call_000001", "content": "/workspace/project"}]
                },
            },
            {"step_id": 3, "source": "agent", "message": "Done."},
        ],
    )

    result = converter.process_row(
        line,
        is_web=False,
        api_env=None,
        api_tool_description="",
        api_sigs={},
    )

    conversations = result["conversations"]
    assert [message["role"] for message in conversations] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert "<function=execute_bash>" in conversations[1]["content"]
    assert conversations[2]["content"].startswith("EXECUTION RESULT of [execute_bash]:")


def test_openhands_v0_converter_preserves_unresolved_xpath_for_web_api(monkeypatch):
    converter = import_openhands_v0_converter(monkeypatch)
    line = atif_line(
        "xpath-preservation",
        [
            {"step_id": 1, "source": "user", "message": "Book a flight."},
            {
                "step_id": 2,
                "source": "agent",
                "message": "",
                "observation": {
                    "results": [
                        {
                            "content": "",
                            "extra": {
                                "web": {
                                    "html": "<html><input id='departure'></html>",
                                    "url": "https://example.com/flights",
                                }
                            },
                        }
                    ]
                },
            },
            {
                "step_id": 3,
                "source": "agent",
                "message": "",
                "tool_calls": [
                    {
                        "tool_call_id": "call_000001",
                        "function_name": "type",
                        "arguments": {"xpath": "//input[@id='departure']", "value": "New York"},
                    }
                ],
            },
        ],
    )

    result = converter.process_row(
        line,
        is_web=True,
        api_env="browser",
        api_tool_description="",
        api_sigs={"type": {"required": ["bid", "value"], "optional": []}},
    )

    browser_call = result["conversations"][1]["content"]
    assert "<function=browser>" in browser_call
    assert "type(" in browser_call
    assert "bid=\"//input[@id='departure']\"" in browser_call


def test_openhands_v0_converter_escapes_function_examples_in_non_tool_messages(monkeypatch):
    converter = import_openhands_v0_converter(monkeypatch)
    line = atif_line(
        "function-example",
        [
            {
                "step_id": 1,
                "source": "user",
                "message": (
                    "Example syntax: <function=execute_bash></function> "
                    "<function_calls> <invoke name=finish>"
                ),
            }
        ],
    )

    result = converter.process_row(
        line,
        is_web=False,
        api_env=None,
        api_tool_description="",
        api_sigs={},
    )

    value = result["conversations"][0]["content"]
    assert result["conversations"][0]["role"] == "user"
    assert "<function=" not in value
    assert "<function_calls>" not in value
    assert "<invoke name=" not in value
    assert "&lt;function=execute_bash>" in value
    assert "&lt;function_calls>" in value
    assert "&lt;invoke name=finish>" in value


def test_openhands_v0_converter_preserves_embedded_function_call_role(monkeypatch):
    converter = import_openhands_v0_converter(monkeypatch)
    line = atif_line(
        "embedded-function-call",
        [
            {"step_id": 1, "source": "user", "message": "Finish the task."},
            {
                "step_id": 2,
                "source": "agent",
                "message": (
                    "<function=finish>\n"
                    "<parameter=message>done</parameter>\n"
                    "<parameter=task_completed>true</parameter>\n"
                    "</function>"
                ),
            },
        ],
    )

    result = converter.process_row(
        line,
        is_web=False,
        api_env=None,
        api_tool_description="",
        api_sigs={},
    )

    assert result["conversations"][1]["role"] == "assistant"
    assert "<function=finish>" in result["conversations"][1]["content"]


def test_openhands_v0_converter_does_not_duplicate_execution_result_prefix(monkeypatch):
    converter = import_openhands_v0_converter(monkeypatch)
    line = atif_line(
        "prefixed-observation",
        [
            {"step_id": 1, "source": "user", "message": "Run pwd."},
            {
                "step_id": 2,
                "source": "agent",
                "message": "",
                "tool_calls": [
                    {
                        "tool_call_id": "call_000001",
                        "function_name": "execute_bash",
                        "arguments": {"command": "pwd"},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "call_000001",
                            "content": "EXECUTION RESULT of [execute_bash]:\n/workspace",
                        }
                    ]
                },
            },
        ],
    )

    result = converter.process_row(
        line,
        is_web=False,
        api_env=None,
        api_tool_description="",
        api_sigs={},
    )

    assert result["conversations"][2]["content"].count("EXECUTION RESULT of") == 1
