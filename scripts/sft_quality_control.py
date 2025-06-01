#!/usr/bin/env python3
"""
Quality control script for SFT data.
Analyzes sample_sft.json files across datasets and generates visualizations.
"""

import glob
import json
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

# Regular expression to match function calls
FUNCTION_PATTERN = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
THOUGHT_PATTERN = re.compile(r"(.*?)<function=", re.DOTALL)


def analyze_dataset(file_path):
    """Analyze a single dataset file and return statistics."""
    with open(file_path, "r") as f:
        data = json.load(f)

    # Extract dataset name from path
    dataset_name = os.path.basename(os.path.dirname(file_path))

    # Initialize counters
    roles = Counter()
    conversation_count = len(data)
    function_calls = 0
    function_names = Counter()
    function_thoughts = 0

    for conversation in data:
        for message in conversation.get("conversations", []):
            role = message.get("from", "unknown")
            content = message.get("value", "")

            # Count roles
            roles[role] += 1

            # Check for function calls
            if role == "function_call":
                match = FUNCTION_PATTERN.search(content)
                if match:
                    function_name = match.group(1)
                    function_calls += 1
                    function_names[function_name] += 1

                    # Check for thoughts before function call
                    thought_match = THOUGHT_PATTERN.search(content)
                    if thought_match and thought_match.group(1).strip():
                        function_thoughts += 1
                else:
                    # Alternative pattern for function calls without </function> closing tag
                    match = re.search(r"<function=([^>]+)>", content)
                    if match:
                        function_name = match.group(1)
                        function_calls += 1
                        function_names[function_name] += 1

    return {
        "dataset": dataset_name,
        "conversation_count": conversation_count,
        "roles": dict(roles),
        "function_calls": function_calls,
        "function_names": dict(function_names),
        "function_thoughts": function_thoughts,
    }


def create_roles_chart(results):
    """Create a stacked bar chart of roles per conversation."""
    datasets = [r["dataset"] for r in results]

    # Get all unique roles across datasets
    all_roles = set()
    for r in results:
        all_roles.update(r["roles"].keys())

    # Create DataFrame for plotting
    data = []
    for r in results:
        row = {"dataset": r["dataset"]}
        conv_count = r["conversation_count"]
        for role in all_roles:
            row[role] = r["roles"].get(role, 0) / conv_count if conv_count > 0 else 0
        data.append(row)

    df = pd.DataFrame(data)
    df = df.set_index("dataset")

    # Save raw data to CSV
    df.to_csv("quality-control-results/roles_per_conversation.csv")

    # Create stacked bar chart
    ax = df.plot(kind="bar", stacked=True, figsize=(12, 8))
    plt.title("Roles per Conversation by Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Average Count per Conversation")
    plt.legend(title="Role")
    # Rotate x-axis labels to 45 degrees
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("quality-control-results/roles_per_conversation.png")
    plt.close()


def create_function_names_chart(results):
    """Create a stacked bar chart of function names."""
    datasets = [r["dataset"] for r in results]

    # Get all unique function names across datasets
    all_functions = set()
    for r in results:
        all_functions.update(r["function_names"].keys())

    # Create DataFrame for plotting
    data = []
    for r in results:
        row = {"dataset": r["dataset"]}
        assistant_msgs = r["roles"].get("function_call", 0)
        for func in all_functions:
            row[func] = (
                r["function_names"].get(func, 0) / assistant_msgs if assistant_msgs > 0 else 0
            )
        data.append(row)

    df = pd.DataFrame(data)
    df = df.set_index("dataset")

    # Save raw data to CSV
    df.to_csv("quality-control-results/function_names.csv")

    # Create stacked bar chart
    ax = df.plot(kind="bar", stacked=True, figsize=(12, 8))
    plt.title("Function Names as Proportion of Assistant Messages")
    plt.xlabel("Dataset")
    plt.ylabel("Proportion")
    plt.legend(title="Function Name", bbox_to_anchor=(1.05, 1), loc="upper left")
    # Rotate x-axis labels to 45 degrees
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("quality-control-results/function_names.png")
    plt.close()


def create_function_thought_chart(results):
    """Create a bar chart of function thoughts percentage."""
    datasets = [r["dataset"] for r in results]
    thought_percentages = []

    # Calculate thought percentages
    data = []
    for r in results:
        thought_pct = (
            r["function_thoughts"] / r["function_calls"] * 100 if r["function_calls"] > 0 else 0
        )
        thought_percentages.append(thought_pct)
        data.append({"dataset": r["dataset"], "thought_percentage": thought_pct})

    # Save raw data to CSV
    df = pd.DataFrame(data)
    df.to_csv("quality-control-results/function_thought_percentage.csv", index=False)

    # Create bar chart
    plt.figure(figsize=(12, 8))
    plt.bar(datasets, thought_percentages)
    plt.title("Percentage of Function Calls with Thoughts")
    plt.xlabel("Dataset")
    plt.ylabel("Percentage (%)")
    # Rotate x-axis labels to 45 degrees
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("quality-control-results/function_thought_percentage.png")
    plt.close()


def main():
    # Create output directory if it doesn't exist
    os.makedirs("quality-control-results", exist_ok=True)

    # Find all sample_sft.json files
    files = glob.glob("datasets/*/sample_sft.json")
    print(f"Found {len(files)} dataset files")

    # Analyze each dataset
    results = []
    for file_path in files:
        print(f"Analyzing {file_path}...")
        result = analyze_dataset(file_path)
        results.append(result)
        print(f"  Conversations: {result['conversation_count']}")
        print(f"  Roles: {result['roles']}")
        print(f"  Function calls: {result['function_calls']}")
        print(f"  Function thoughts: {result['function_thoughts']}")

    # Create visualizations
    create_roles_chart(results)
    create_function_names_chart(results)
    create_function_thought_chart(results)

    print("\nAnalysis complete. Generated files in quality-control-results/:")
    print("- roles_per_conversation.png and roles_per_conversation.csv")
    print("- function_names.png and function_names.csv")
    print("- function_thought_percentage.png and function_thought_percentage.csv")


if __name__ == "__main__":
    main()
