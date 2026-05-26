import importlib.util
from pathlib import Path

from schema.action.code import CodeAction
from schema.action.message import MessageAction

RAW_TO_STANDARDIZED_PATH = (
    Path(__file__).parent.parent / "datasets" / "agenttuning_db" / "raw_to_standardized.py"
)


def load_converter_module():
    spec = importlib.util.spec_from_file_location(
        "agenttuning_db_raw_to_standardized", RAW_TO_STANDARDIZED_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sql_final_answer_converts_to_code_action():
    converter = load_converter_module()

    trajectory = converter.convert_trajectory(
        {
            "id": "db_sql_final_answer",
            "conversations": [
                {
                    "role": "user",
                    "content": "Insert a row into Employees.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "Inserted the employee.\nAction: Answer\nFinal Answer: "
                        "[\"INSERT INTO Employees (first_name) VALUES ('Emily')\"]"
                    ),
                },
            ],
        }
    )

    action = trajectory.content[-1]
    assert isinstance(action, CodeAction)
    assert action.language == "mysql"
    assert action.content == "INSERT INTO Employees (first_name) VALUES ('Emily')"
    assert action.description == "Inserted the employee.\n"


def test_non_sql_final_answer_stays_finish_action():
    converter = load_converter_module()

    trajectory = converter.convert_trajectory(
        {
            "id": "db_non_sql_final_answer",
            "conversations": [
                {
                    "role": "user",
                    "content": "Insert a row into Books.",
                },
                {
                    "role": "assistant",
                    "content": 'Inserted the book.\nAction: Answer\nFinal Answer: ["none"]',
                },
            ],
        }
    )

    action = trajectory.content[-1]
    assert isinstance(action, MessageAction)
    assert action.content == '<finish> <solution> ["none"] </solution> </finish>'
