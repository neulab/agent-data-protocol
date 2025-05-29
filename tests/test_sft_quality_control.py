#!/usr/bin/env python3
"""
Tests for the SFT quality control script.
"""

import os
import json
import tempfile
import shutil
import pytest
import sys
import csv

# Add the parent directory to the path so we can import the script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.sft_quality_control import (
    extract_function_calls,
    has_thought,
    analyze_sft_data,
    find_sft_files
)


@pytest.fixture
def test_environment():
    """Set up test environment."""
    # Create a temporary directory
    test_dir = tempfile.mkdtemp()
    
    # Create test datasets
    dataset1_dir = os.path.join(test_dir, 'dataset1')
    dataset2_dir = os.path.join(test_dir, 'dataset2')
    os.makedirs(dataset1_dir, exist_ok=True)
    os.makedirs(dataset2_dir, exist_ok=True)
    
    # Create test SFT files
    # Dataset 1 - with thoughts
    dataset1_data = [
        {
            "id": "test1",
            "conversations": [
                {"role": "system", "content": "System message"},
                {"role": "user", "content": "User message"},
                {"role": "assistant", "content": "THOUGHT: I need to execute a command\n\nACTION: \n```execute_bash(command='ls -la')```\n"},
                {"role": "user", "content": "Another user message"},
                {"role": "assistant", "content": "THOUGHT: I'll use Python\n\nACTION: <execute_ipython>\nprint('Hello')\n</execute_ipython>"}
            ]
        },
        {
            "id": "test2",
            "conversations": [
                {"role": "system", "content": "System message"},
                {"role": "user", "content": "User message"},
                {"role": "assistant", "content": "THOUGHT: I'll check the file\n\nACTION: \n```click(bid='123')```\n"}
            ]
        }
    ]
    
    # Dataset 2 - without thoughts
    dataset2_data = [
        {
            "id": "test3",
            "conversations": [
                {"role": "user", "content": "User message"},
                {"role": "assistant", "content": "ACTION: \n```execute_bash(command='echo hello')```\n"},
                {"role": "user", "content": "Another message"},
                {"role": "assistant", "content": "ACTION: \n```click(bid='456')```\n"}
            ]
        }
    ]
    
    # Write to files
    with open(os.path.join(dataset1_dir, 'test_sft.jsonl'), 'w') as f:
        for item in dataset1_data:
            f.write(json.dumps(item) + '\n')
    
    with open(os.path.join(dataset2_dir, 'test_sft.jsonl'), 'w') as f:
        for item in dataset2_data:
            f.write(json.dumps(item) + '\n')
    
    # Return the test directories
    yield {
        'test_dir': test_dir,
        'dataset1_dir': dataset1_dir,
        'dataset2_dir': dataset2_dir
    }
    
    # Clean up after the test
    shutil.rmtree(test_dir)


def test_extract_function_calls():
    """Test extracting function calls from content."""
    # Test with code block
    content1 = "ACTION: \n```execute_bash(command='ls -la')```\n"
    assert extract_function_calls(content1) == ['execute_bash']
    
    # Test with execute tag
    content2 = "ACTION: <execute_ipython>\nprint('Hello')\n</execute_ipython>"
    assert extract_function_calls(content2) == ['ipython']
    
    # Test with multiple function calls
    content3 = "ACTION: \n```click(bid='123')```\n\nACTION: \n```hover(bid='456')```\n"
    assert extract_function_calls(content3) == ['click', 'hover']
    
    # Test with no function calls
    content4 = "This is a regular message with no function calls."
    assert extract_function_calls(content4) == []


def test_has_thought():
    """Test checking if content has a thought section."""
    # Test with thought
    content1 = "THOUGHT: I need to execute a command\n\nACTION: \n```execute_bash(command='ls -la')```\n"
    assert has_thought(content1) is True
    
    # Test with lowercase thought
    content2 = "thought: I'll use Python\n\nACTION: <execute_ipython>\nprint('Hello')\n</execute_ipython>"
    assert has_thought(content2) is True
    
    # Test without thought
    content3 = "ACTION: \n```click(bid='123')```\n"
    assert has_thought(content3) is False


def test_analyze_sft_data(test_environment):
    """Test analyzing SFT data."""
    # Analyze dataset1
    results1 = analyze_sft_data(os.path.join(test_environment['dataset1_dir'], 'test_sft.jsonl'))
    
    # Check conversation stats
    assert len(results1['conversation_stats']) == 2
    assert results1['conversation_stats'][0]['dataset'] == 'dataset1'
    assert results1['conversation_stats'][0]['system_turns'] == 1
    assert results1['conversation_stats'][0]['user_turns'] == 2
    assert results1['conversation_stats'][0]['assistant_turns'] == 2
    
    # Check function calls
    assert len(results1['function_calls_per_turn']) == 3
    assert results1['total_function_calls'] == 3
    assert results1['function_calls_without_thought'] == 0
    assert results1['thought_percentage'] == 0
    
    # Analyze dataset2
    results2 = analyze_sft_data(os.path.join(test_environment['dataset2_dir'], 'test_sft.jsonl'))
    
    # Check conversation stats
    assert len(results2['conversation_stats']) == 1
    assert results2['conversation_stats'][0]['dataset'] == 'dataset2'
    assert results2['conversation_stats'][0]['system_turns'] == 0
    assert results2['conversation_stats'][0]['user_turns'] == 2
    assert results2['conversation_stats'][0]['assistant_turns'] == 2
    
    # Check function calls
    assert len(results2['function_calls_per_turn']) == 2
    assert results2['total_function_calls'] == 2
    assert results2['function_calls_without_thought'] == 2
    assert results2['thought_percentage'] == 100.0


def test_find_sft_files(test_environment):
    """Test finding SFT files."""
    # Create a JSON file to test both extensions
    with open(os.path.join(test_environment['dataset1_dir'], 'test_sft.json'), 'w') as f:
        f.write('{"test": "data"}')
    
    # Find all SFT files with both extensions
    sft_files = find_sft_files([test_environment['test_dir']], '*_sft.{json,jsonl}')
    assert len(sft_files) == 3  # 2 jsonl + 1 json
    
    # Find files with specific pattern
    sft_files = find_sft_files([test_environment['dataset1_dir']], '*_sft.{json,jsonl}')
    assert len(sft_files) == 2  # 1 jsonl + 1 json
    
    # Test with just jsonl pattern
    sft_files = find_sft_files([test_environment['dataset1_dir']], '*_sft.jsonl')
    assert len(sft_files) == 1
    assert sft_files[0].endswith('dataset1/test_sft.jsonl')