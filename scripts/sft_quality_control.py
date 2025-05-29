#!/usr/bin/env python3
"""
SFT Data Quality Control Script

This script analyzes SFT data and generates:
1. A stacked bar chart of function calls per turn by dataset
2. A stacked bar chart of role+turn per conversation by dataset
3. A bar chart of % of function calls w/o thoughts
4. A CSV file with all the relevant data
"""

import argparse
import json
import os
import re
import csv
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate quality control metrics for SFT data')
    parser.add_argument('--input_dirs', nargs='+', required=True,
                        help='List of dataset directories to analyze')
    parser.add_argument('--output_dir', type=str, default='/workspace/quality_control_results',
                        help='Directory to save the output charts and CSV')
    parser.add_argument('--sft_file_pattern', type=str, default='*_sft.{json,jsonl}',
                        help='Pattern to match SFT files (default: *_sft.{json,jsonl})')
    return parser.parse_args()


def extract_function_calls(content):
    """Extract function calls from the content."""
    # Match patterns like ACTION: \n```function_name(args)```\n or <execute_ipython>\n...\n</execute_ipython>
    function_calls = []
    
    # Pattern for code blocks with function calls
    code_block_pattern = r"```([^`]+)```"
    code_blocks = re.findall(code_block_pattern, content)
    for block in code_blocks:
        # Extract function name (assuming it's at the beginning of the line)
        function_match = re.search(r"^(\w+)\(", block.strip())
        if function_match:
            function_calls.append(function_match.group(1))
    
    # Pattern for execute tags
    execute_pattern = r"<execute_(\w+)>"
    execute_matches = re.findall(execute_pattern, content)
    function_calls.extend(execute_matches)
    
    return function_calls


def has_thought(content):
    """Check if the content has a thought section."""
    return "THOUGHT:" in content or "thought:" in content.lower()


def analyze_sft_data(file_path):
    """Analyze a single SFT data file."""
    dataset_name = os.path.basename(os.path.dirname(file_path))
    
    # Initialize counters
    conversation_stats = []
    function_calls_per_turn = []
    function_calls_without_thought = 0
    total_function_calls = 0
    
    # Check if the file is JSON or JSONL
    is_jsonl = file_path.endswith('.jsonl')
    
    try:
        if is_jsonl:
            # Process JSONL file (one JSON object per line)
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        calls, without_thought = process_conversation(
                            data, dataset_name, conversation_stats, 
                            function_calls_per_turn, function_calls_without_thought,
                            total_function_calls
                        )
                        total_function_calls += calls
                        function_calls_without_thought += without_thought
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse line in {file_path}")
                        continue
        else:
            # Process JSON file (array of objects or single object)
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        # Array of conversations
                        for conv in data:
                            calls, without_thought = process_conversation(
                                conv, dataset_name, conversation_stats, 
                                function_calls_per_turn, function_calls_without_thought,
                                total_function_calls
                            )
                            total_function_calls += calls
                            function_calls_without_thought += without_thought
                    else:
                        # Single conversation
                        calls, without_thought = process_conversation(
                            data, dataset_name, conversation_stats, 
                            function_calls_per_turn, function_calls_without_thought,
                            total_function_calls
                        )
                        total_function_calls += calls
                        function_calls_without_thought += without_thought
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON in {file_path}")
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
    
    # Calculate percentage of function calls without thought
    thought_percentage = 0
    if total_function_calls > 0:
        thought_percentage = (function_calls_without_thought / total_function_calls) * 100
    
    return {
        'conversation_stats': conversation_stats,
        'function_calls_per_turn': function_calls_per_turn,
        'function_calls_without_thought': function_calls_without_thought,
        'total_function_calls': total_function_calls,
        'thought_percentage': thought_percentage
    }


def process_conversation(data, dataset_name, conversation_stats, function_calls_per_turn, 
                         function_calls_without_thought, total_function_calls):
    """Process a single conversation data object."""
    # Count roles per conversation
    role_counts = Counter()
    function_calls_in_conv = defaultdict(int)
    
    # Track function calls and thoughts for this conversation
    conv_function_calls = 0
    conv_calls_without_thought = 0
    
    for turn in data.get('conversations', []):
        role = turn.get('role', '')
        role_counts[role] += 1
        
        # Extract function calls
        content = turn.get('content', '')
        if isinstance(content, list):
            content = ' '.join([item.get('text', '') for item in content if item.get('type') == 'text'])
        
        if role == 'assistant':
            function_calls = extract_function_calls(content)
            for func in function_calls:
                function_calls_in_conv[func] += 1
                conv_function_calls += 1
                
                # Check if there's a thought
                if not has_thought(content):
                    conv_calls_without_thought += 1
    
    # Store conversation stats
    conversation_stats.append({
        'dataset': dataset_name,
        'id': data.get('id', ''),
        'system_turns': role_counts.get('system', 0),
        'user_turns': role_counts.get('user', 0),
        'assistant_turns': role_counts.get('assistant', 0),
        'total_turns': sum(role_counts.values())
    })
    
    # Store function calls per turn
    if role_counts.get('assistant', 0) > 0:
        for func, count in function_calls_in_conv.items():
            function_calls_per_turn.append({
                'dataset': dataset_name,
                'id': data.get('id', ''),
                'function': func,
                'count': count,
                'per_turn': count / role_counts.get('assistant', 1)
            })
            
    # Return the counts for this conversation
    return conv_function_calls, conv_calls_without_thought


def find_sft_files(input_dirs, pattern):
    """Find all SFT files in the input directories."""
    import glob
    sft_files = set()  # Use a set to avoid duplicates
    
    for input_dir in input_dirs:
        for root, _, files in os.walk(input_dir):
            # If the pattern contains {json,jsonl}, handle both extensions
            if '{json,jsonl}' in pattern:
                for ext in ['json', 'jsonl']:
                    # Replace {json,jsonl} with the current extension
                    current_pattern = pattern.replace('{json,jsonl}', ext)
                    # Use glob to match the pattern
                    matches = glob.glob(os.path.join(root, current_pattern))
                    sft_files.update(matches)
            else:
                # Use the pattern as is
                matches = glob.glob(os.path.join(root, pattern))
                sft_files.update(matches)
                
    return sorted(list(sft_files))  # Convert back to sorted list


def generate_charts(all_results, output_dir):
    """Generate charts from the analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for charts
    datasets = []
    function_calls_data = defaultdict(lambda: defaultdict(float))
    role_turns_data = defaultdict(lambda: defaultdict(float))
    thought_percentages = []
    
    for dataset, results in all_results.items():
        datasets.append(dataset)
        
        # Function calls per turn
        function_counts = defaultdict(float)
        for item in results['function_calls_per_turn']:
            function_counts[item['function']] += item['per_turn']
        
        # Average function calls per turn
        total_assistant_turns = sum(item['assistant_turns'] for item in results['conversation_stats'])
        if total_assistant_turns > 0:
            for func, count in function_counts.items():
                function_calls_data[dataset][func] = count / len(results['conversation_stats'])
        
        # Role turns per conversation
        total_conversations = len(results['conversation_stats'])
        if total_conversations > 0:
            role_turns_data[dataset]['system'] = sum(item['system_turns'] for item in results['conversation_stats']) / total_conversations
            role_turns_data[dataset]['user'] = sum(item['user_turns'] for item in results['conversation_stats']) / total_conversations
            role_turns_data[dataset]['assistant'] = sum(item['assistant_turns'] for item in results['conversation_stats']) / total_conversations
        
        # Thought percentage
        thought_percentages.append({
            'dataset': dataset,
            'percentage': results['thought_percentage']
        })
    
    # 1. Stacked bar chart of function calls per turn by dataset
    plt.figure(figsize=(12, 8))
    
    # Get all unique functions
    all_functions = set()
    for dataset_data in function_calls_data.values():
        all_functions.update(dataset_data.keys())
    
    # Create the stacked bar chart
    bottom = np.zeros(len(datasets))
    for function in all_functions:
        values = [function_calls_data[dataset].get(function, 0) for dataset in datasets]
        plt.bar(datasets, values, bottom=bottom, label=function)
        bottom += values
    
    plt.xlabel('Dataset')
    plt.ylabel('Function Calls per Turn')
    plt.title('Function Calls per Turn by Dataset')
    plt.legend(title='Function Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'function_calls_per_turn.png'))
    plt.close()
    
    # 2. Stacked bar chart of role+turn per conversation by dataset
    plt.figure(figsize=(12, 8))
    
    # Create the stacked bar chart
    bottom = np.zeros(len(datasets))
    for role in ['system', 'user', 'assistant']:
        values = [role_turns_data[dataset].get(role, 0) for dataset in datasets]
        plt.bar(datasets, values, bottom=bottom, label=role)
        bottom += values
    
    plt.xlabel('Dataset')
    plt.ylabel('Turns per Conversation')
    plt.title('Role Turns per Conversation by Dataset')
    plt.legend(title='Role', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'role_turns_per_conversation.png'))
    plt.close()
    
    # 3. Bar chart of % of function calls w/o thoughts
    plt.figure(figsize=(12, 8))
    
    datasets_with_thoughts = [item['dataset'] for item in thought_percentages]
    percentages = [item['percentage'] for item in thought_percentages]
    
    plt.bar(datasets_with_thoughts, percentages)
    plt.xlabel('Dataset')
    plt.ylabel('Percentage (%)')
    plt.title('Percentage of Function Calls Without Thoughts')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'function_calls_without_thoughts.png'))
    plt.close()


def write_csv(all_results, output_dir):
    """Write all the data to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for CSV
    csv_data = []
    
    for dataset, results in all_results.items():
        # Function calls per turn
        function_counts = defaultdict(float)
        for item in results['function_calls_per_turn']:
            function_counts[item['function']] += item['per_turn']
        
        # Average function calls per turn
        total_assistant_turns = sum(item['assistant_turns'] for item in results['conversation_stats'])
        avg_function_calls = {}
        if total_assistant_turns > 0:
            for func, count in function_counts.items():
                avg_function_calls[func] = count / len(results['conversation_stats'])
        
        # Role turns per conversation
        total_conversations = len(results['conversation_stats'])
        avg_role_turns = {}
        if total_conversations > 0:
            avg_role_turns['system'] = sum(item['system_turns'] for item in results['conversation_stats']) / total_conversations
            avg_role_turns['user'] = sum(item['user_turns'] for item in results['conversation_stats']) / total_conversations
            avg_role_turns['assistant'] = sum(item['assistant_turns'] for item in results['conversation_stats']) / total_conversations
        
        # Add to CSV data
        csv_data.append({
            'dataset': dataset,
            'total_conversations': total_conversations,
            'total_function_calls': results['total_function_calls'],
            'function_calls_without_thought': results['function_calls_without_thought'],
            'thought_percentage': results['thought_percentage'],
            'avg_system_turns': avg_role_turns.get('system', 0),
            'avg_user_turns': avg_role_turns.get('user', 0),
            'avg_assistant_turns': avg_role_turns.get('assistant', 0),
            **{f'avg_{func}_calls': count for func, count in avg_function_calls.items()}
        })
    
    # Write to CSV
    if csv_data:
        # Get all field names
        fieldnames = set()
        for item in csv_data:
            fieldnames.update(item.keys())
        
        with open(os.path.join(output_dir, 'sft_quality_metrics.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
            writer.writeheader()
            writer.writerows(csv_data)


def main():
    """Main function."""
    args = parse_args()
    
    # Find all SFT files
    sft_files = find_sft_files(args.input_dirs, args.sft_file_pattern)
    
    if not sft_files:
        print(f"No SFT files found matching pattern '{args.sft_file_pattern}' in the specified directories.")
        return
    
    print(f"Found {len(sft_files)} SFT files to analyze.")
    
    # Analyze each file
    all_results = defaultdict(lambda: {
        'conversation_stats': [],
        'function_calls_per_turn': [],
        'function_calls_without_thought': 0,
        'total_function_calls': 0,
        'thought_percentage': 0
    })
    
    for file_path in sft_files:
        print(f"Analyzing {file_path}...")
        dataset_name = os.path.basename(os.path.dirname(file_path))
        results = analyze_sft_data(file_path)
        
        # Merge results
        all_results[dataset_name]['conversation_stats'].extend(results['conversation_stats'])
        all_results[dataset_name]['function_calls_per_turn'].extend(results['function_calls_per_turn'])
        all_results[dataset_name]['function_calls_without_thought'] += results['function_calls_without_thought']
        all_results[dataset_name]['total_function_calls'] += results['total_function_calls']
        
        # Recalculate thought percentage
        if all_results[dataset_name]['total_function_calls'] > 0:
            all_results[dataset_name]['thought_percentage'] = (
                all_results[dataset_name]['function_calls_without_thought'] / 
                all_results[dataset_name]['total_function_calls'] * 100
            )
    
    # Generate charts
    generate_charts(all_results, args.output_dir)
    
    # Write CSV
    write_csv(all_results, args.output_dir)
    
    print(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()