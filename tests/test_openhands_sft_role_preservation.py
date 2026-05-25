import importlib
import json
import sys
import types


def import_openhands_converter(monkeypatch):
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

    fake_system_module = types.ModuleType("agents.openhands.system_prompt.system")
    fake_system_module.get_system_message = lambda *args, **kwargs: "system prompt"
    monkeypatch.setitem(sys.modules, "agents.openhands.system_prompt.system", fake_system_module)

    fake_user_module = types.ModuleType("agents.openhands.system_prompt.user")
    fake_user_module.get_web_user_message = lambda *args, **kwargs: "web prompt"
    monkeypatch.setitem(sys.modules, "agents.openhands.system_prompt.user", fake_user_module)

    monkeypatch.delitem(sys.modules, "agents.openhands.std_to_sft", raising=False)
    return importlib.import_module("agents.openhands.std_to_sft")


def test_openhands_converter_preserves_tool_roles(monkeypatch):
    converter = import_openhands_converter(monkeypatch)
    line = json.dumps(
        {
            "id": "role-preservation",
            "content": [
                {
                    "class_": "text_observation",
                    "content": "Run pwd.",
                    "source": "user",
                },
                {
                    "class_": "code_action",
                    "language": "bash",
                    "content": "pwd",
                    "description": None,
                },
                {
                    "class_": "text_observation",
                    "content": "/workspace/project",
                    "source": "environment",
                },
                {
                    "class_": "message_action",
                    "content": "Done.",
                    "description": None,
                },
            ],
        }
    )

    result = converter.process_row(
        line,
        is_web=False,
        api_env=None,
        api_tool_description="",
        api_sigs={},
    )

    conversations = result["conversations"]
    assert [message["from"] for message in conversations] == [
        "human",
        "function_call",
        "observation",
        "gpt",
    ]
    assert "<function=execute_bash>" in conversations[1]["value"]
    assert conversations[2]["value"].startswith("EXECUTION RESULT of [execute_bash]:")


def test_openhands_converter_preserves_unresolved_xpath_for_web_api(monkeypatch):
    converter = import_openhands_converter(monkeypatch)
    line = json.dumps(
        {
            "id": "xpath-preservation",
            "content": [
                {
                    "class_": "text_observation",
                    "content": "Book a flight.",
                    "source": "user",
                },
                {
                    "class_": "web_observation",
                    "html": "<html><input id='departure'></html>",
                    "axtree": None,
                    "url": "https://example.com/flights",
                    "image_observation": None,
                    "viewport_size": None,
                },
                {
                    "class_": "api_action",
                    "function": "type",
                    "kwargs": {"xpath": "//input[@id='departure']", "value": "New York"},
                },
            ],
        }
    )

    result = converter.process_row(
        line,
        is_web=True,
        api_env="browser",
        api_tool_description="",
        api_sigs={"type": {"required": ["bid", "value"], "optional": []}},
    )

    browser_call = result["conversations"][1]["value"]
    assert "<function=browser>" in browser_call
    assert "type(" in browser_call
    assert "bid=\"//input[@id='departure']\"" in browser_call


def test_openhands_converter_escapes_function_examples_in_non_tool_messages(monkeypatch):
    converter = import_openhands_converter(monkeypatch)
    line = json.dumps(
        {
            "id": "function-example",
            "content": [
                {
                    "class_": "text_observation",
                    "content": (
                        "Example syntax: <function=execute_bash></function> "
                        "<function_calls> <invoke name=finish>"
                    ),
                    "source": "user",
                },
            ],
        }
    )

    result = converter.process_row(
        line,
        is_web=False,
        api_env=None,
        api_tool_description="",
        api_sigs={},
    )

    value = result["conversations"][0]["value"]
    assert result["conversations"][0]["from"] == "human"
    assert "<function=" not in value
    assert "<function_calls>" not in value
    assert "<invoke name=" not in value
    assert "&lt;function=execute_bash>" in value
    assert "&lt;function_calls>" in value
    assert "&lt;invoke name=finish>" in value


def test_openhands_converter_preserves_embedded_function_call_role(monkeypatch):
    converter = import_openhands_converter(monkeypatch)
    line = json.dumps(
        {
            "id": "embedded-function-call",
            "content": [
                {
                    "class_": "text_observation",
                    "content": "Finish the task.",
                    "source": "user",
                },
                {
                    "class_": "message_action",
                    "content": (
                        "<function=finish>\n"
                        "<parameter=message>done</parameter>\n"
                        "<parameter=task_completed>true</parameter>\n"
                        "</function>"
                    ),
                    "description": None,
                },
            ],
        }
    )

    result = converter.process_row(
        line,
        is_web=False,
        api_env=None,
        api_tool_description="",
        api_sigs={},
    )

    assert result["conversations"][1]["from"] == "function_call"
    assert "<function=finish>" in result["conversations"][1]["value"]


def test_openhands_converter_does_not_duplicate_execution_result_prefix(monkeypatch):
    converter = import_openhands_converter(monkeypatch)
    line = json.dumps(
        {
            "id": "prefixed-observation",
            "content": [
                {
                    "class_": "text_observation",
                    "content": "Run pwd.",
                    "source": "user",
                },
                {
                    "class_": "code_action",
                    "language": "bash",
                    "content": "pwd",
                    "description": None,
                },
                {
                    "class_": "text_observation",
                    "content": "EXECUTION RESULT of [execute_bash]:\n/workspace",
                    "source": "environment",
                },
            ],
        }
    )

    result = converter.process_row(
        line,
        is_web=False,
        api_env=None,
        api_tool_description="",
        api_sigs={},
    )

    assert result["conversations"][2]["value"].count("EXECUTION RESULT of") == 1
