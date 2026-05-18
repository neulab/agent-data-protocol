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
    fake_system_module.get_system_message = lambda: "system prompt"
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
