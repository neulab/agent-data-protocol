import json
import os
from unittest.mock import patch

import pytest

from scripts.sft_quality_control import (
    analyze_dataset,
    create_function_names_chart,
    create_function_thought_chart,
    create_roles_chart,
)


@pytest.fixture
def sample_data():
    """Create a sample dataset for testing."""
    return [
        {
            "conversations": [
                {"from": "human", "value": "Hello, can you help me?"},
                {"from": "assistant", "value": "I'll help you with that."},
                {
                    "from": "function_call",
                    "value": "I need to search for information.<function=search>query=test</function>",
                },
                {"from": "function_result", "value": "Search results for test"},
                {"from": "assistant", "value": "Here are the results."},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "Another question"},
                {"from": "assistant", "value": "Let me think about that."},
                {
                    "from": "function_call",
                    "value": "<function=calculate>1+1</function>",
                },
                {"from": "function_result", "value": "2"},
                {
                    "from": "function_call",
                    "value": "Let me think about this more.<function=search>query=another test</function>",
                },
            ]
        },
    ]


def test_analyze_dataset(sample_data, tmp_path):
    """Test the analyze_dataset function."""
    # Create a temporary dataset file
    dataset_dir = tmp_path / "test_dataset"
    dataset_dir.mkdir()
    dataset_file = dataset_dir / "sample_sft.json"

    with open(dataset_file, "w") as f:
        json.dump(sample_data, f)

    # Analyze the dataset
    result = analyze_dataset(str(dataset_file))

    # Check the results
    assert result["dataset"] == "test_dataset"
    assert result["conversation_count"] == 2
    assert result["roles"]["human"] == 2
    assert result["roles"]["assistant"] == 3
    assert result["roles"]["function_call"] == 3
    assert result["roles"]["function_result"] == 2
    assert result["function_calls"] == 3
    assert result["function_names"]["search"] == 2
    assert result["function_names"]["calculate"] == 1
    assert result["function_thoughts"] == 2  # Two function calls have thoughts before them


@pytest.mark.parametrize(
    "chart_function",
    [create_roles_chart, create_function_names_chart, create_function_thought_chart],
)
def test_chart_creation(chart_function, tmp_path, monkeypatch):
    """Test that chart creation functions run without errors."""
    # Mock the results
    results = [
        {
            "dataset": "test_dataset",
            "conversation_count": 2,
            "roles": {"human": 2, "assistant": 3, "function_call": 3, "function_result": 2},
            "function_calls": 3,
            "function_names": {"search": 2, "calculate": 1},
            "function_thoughts": 1,
        }
    ]

    # Change to the temporary directory
    monkeypatch.chdir(tmp_path)

    # Create the output directory
    os.makedirs("quality-control-results", exist_ok=True)

    # Run the chart function
    with patch("matplotlib.pyplot.savefig"):  # Mock savefig to avoid actual file creation
        chart_function(results)

    # Check that CSV files are created
    if chart_function == create_roles_chart:
        assert os.path.exists("quality-control-results/roles_per_conversation.csv")
    elif chart_function == create_function_names_chart:
        assert os.path.exists("quality-control-results/function_names.csv")
    elif chart_function == create_function_thought_chart:
        assert os.path.exists("quality-control-results/function_thought_percentage.csv")


def test_function_pattern_matching():
    """Test the regular expression patterns for function calls."""
    from scripts.sft_quality_control import FUNCTION_PATTERN, THOUGHT_PATTERN

    # Test function pattern with content
    content = "<function=search>query=test</function>"
    match = FUNCTION_PATTERN.search(content)
    assert match
    assert match.group(1) == "search"
    assert match.group(2) == "query=test"

    # Test function pattern without closing tag
    content = "<function=search>query=test"
    match = FUNCTION_PATTERN.search(content)
    assert not match  # Should not match without closing tag

    # Test thought pattern
    content = "I need to search for information.<function=search>query=test</function>"
    match = THOUGHT_PATTERN.search(content)
    assert match
    assert match.group(1).strip() == "I need to search for information."

    # Test thought pattern with no thought
    content = "<function=search>query=test</function>"
    match = THOUGHT_PATTERN.search(content)
    assert match
    assert match.group(1).strip() == ""

    # Test orca_agentinstruct style function pattern
    content = """<function=finish>
<parameter=message>Test response</parameter>
<parameter=task_completed>true</parameter>
</function>"""
    match = FUNCTION_PATTERN.search(content)
    assert match
    assert match.group(1) == "finish"


def test_analyze_dataset_with_gpt_role_functions(tmp_path):
    """Test analyzing a dataset with function calls in 'gpt' role messages."""
    # Create a sample dataset with function calls in 'gpt' role
    sample_data = [
        {
            "conversations": [
                {"from": "human", "value": "Test question"},
                {
                    "from": "gpt",
                    "value": "<function=finish>\n<parameter=message>Test response</parameter>\n<parameter=task_completed>true</parameter>\n</function>",
                },
            ]
        }
    ]

    # Create a temporary dataset file
    dataset_dir = tmp_path / "test_gpt_role"
    dataset_dir.mkdir()
    dataset_file = dataset_dir / "sample_sft.json"

    with open(dataset_file, "w") as f:
        json.dump(sample_data, f)

    # Analyze the dataset
    result = analyze_dataset(str(dataset_file))

    # Check that the function call was detected
    assert result["function_calls"] == 1, "Should detect 1 function call in 'gpt' role message"
    assert result["function_names"]["finish"] == 1, (
        "Should detect 'finish' function in 'gpt' role message"
    )
