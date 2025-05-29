#!/usr/bin/env python3
"""
Tests for the SFT quality control script.
"""

import os
import json
import tempfile
import shutil
import unittest
from unittest.mock import patch
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


class TestSFTQualityControl(unittest.TestCase):
    """Test cases for SFT quality control script."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        
        # Create test datasets
        self.dataset1_dir = os.path.join(self.test_dir, 'dataset1')
        self.dataset2_dir = os.path.join(self.test_dir, 'dataset2')
        os.makedirs(self.dataset1_dir, exist_ok=True)
        os.makedirs(self.dataset2_dir, exist_ok=True)
        
        # Create test SFT files
        self.create_test_sft_files()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)

    def create_test_sft_files(self):
        """Create test SFT files."""
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
        with open(os.path.join(self.dataset1_dir, 'test_sft.jsonl'), 'w') as f:
            for item in dataset1_data:
                f.write(json.dumps(item) + '\n')
        
        with open(os.path.join(self.dataset2_dir, 'test_sft.jsonl'), 'w') as f:
            for item in dataset2_data:
                f.write(json.dumps(item) + '\n')

    def test_extract_function_calls(self):
        """Test extracting function calls from content."""
        # Test with code block
        content1 = "ACTION: \n```execute_bash(command='ls -la')```\n"
        self.assertEqual(extract_function_calls(content1), ['execute_bash'])
        
        # Test with execute tag
        content2 = "ACTION: <execute_ipython>\nprint('Hello')\n</execute_ipython>"
        self.assertEqual(extract_function_calls(content2), ['ipython'])
        
        # Test with multiple function calls
        content3 = "ACTION: \n```click(bid='123')```\n\nACTION: \n```hover(bid='456')```\n"
        self.assertEqual(extract_function_calls(content3), ['click', 'hover'])
        
        # Test with no function calls
        content4 = "This is a regular message with no function calls."
        self.assertEqual(extract_function_calls(content4), [])

    def test_has_thought(self):
        """Test checking if content has a thought section."""
        # Test with thought
        content1 = "THOUGHT: I need to execute a command\n\nACTION: \n```execute_bash(command='ls -la')```\n"
        self.assertTrue(has_thought(content1))
        
        # Test with lowercase thought
        content2 = "thought: I'll use Python\n\nACTION: <execute_ipython>\nprint('Hello')\n</execute_ipython>"
        self.assertTrue(has_thought(content2))
        
        # Test without thought
        content3 = "ACTION: \n```click(bid='123')```\n"
        self.assertFalse(has_thought(content3))

    def test_analyze_sft_data(self):
        """Test analyzing SFT data."""
        # Analyze dataset1
        results1 = analyze_sft_data(os.path.join(self.dataset1_dir, 'test_sft.jsonl'))
        
        # Check conversation stats
        self.assertEqual(len(results1['conversation_stats']), 2)
        self.assertEqual(results1['conversation_stats'][0]['dataset'], 'dataset1')
        self.assertEqual(results1['conversation_stats'][0]['system_turns'], 1)
        self.assertEqual(results1['conversation_stats'][0]['user_turns'], 2)
        self.assertEqual(results1['conversation_stats'][0]['assistant_turns'], 2)
        
        # Check function calls
        self.assertEqual(len(results1['function_calls_per_turn']), 3)
        self.assertEqual(results1['total_function_calls'], 3)
        self.assertEqual(results1['function_calls_without_thought'], 0)
        self.assertEqual(results1['thought_percentage'], 0)
        
        # Analyze dataset2
        results2 = analyze_sft_data(os.path.join(self.dataset2_dir, 'test_sft.jsonl'))
        
        # Check conversation stats
        self.assertEqual(len(results2['conversation_stats']), 1)
        self.assertEqual(results2['conversation_stats'][0]['dataset'], 'dataset2')
        self.assertEqual(results2['conversation_stats'][0]['system_turns'], 0)
        self.assertEqual(results2['conversation_stats'][0]['user_turns'], 2)
        self.assertEqual(results2['conversation_stats'][0]['assistant_turns'], 2)
        
        # Check function calls
        self.assertEqual(len(results2['function_calls_per_turn']), 2)
        self.assertEqual(results2['total_function_calls'], 2)
        self.assertEqual(results2['function_calls_without_thought'], 2)
        self.assertEqual(results2['thought_percentage'], 100.0)

    def test_find_sft_files(self):
        """Test finding SFT files."""
        # Find all SFT files
        sft_files = find_sft_files([self.test_dir], '*_sft.jsonl')
        self.assertEqual(len(sft_files), 2)
        
        # Find files with specific pattern
        sft_files = find_sft_files([self.dataset1_dir], '*_sft.jsonl')
        self.assertEqual(len(sft_files), 1)
        self.assertTrue(sft_files[0].endswith('dataset1/test_sft.jsonl'))


if __name__ == '__main__':
    unittest.main()