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

# List of valid tools from the system_prompt/tools directory
VALID_TOOLS = [
    "execute_bash",
    "think",
    "finish",
    "browser",
    "execute_ipython_cell",
    "str_replace_editor",
]


def analyze_dataset(file_path):
    """Analyze a single dataset file and return statistics."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract dataset name from path
    dataset_name = os.path.basename(os.path.dirname(file_path))

    # Initialize counters
    roles = Counter()
    conversation_count = len(data)
    function_calls = 0
    function_names = Counter()
    function_thoughts = 0
    finish_action_count = 0
    len1_conversation_count = 0
    invalid_tools = set()

    for conversation in data:
        conversation_has_finish = False
        if len(conversation.get("conversations", [])) == 2:
            len1_conversation_count += 1
        for message in conversation.get("conversations", []):
            role = message.get("from", "unknown")
            content = message.get("value", "")

            # Count roles
            roles[role] += 1

            # Check for function calls in any message content
            match = FUNCTION_PATTERN.search(content)
            if match:
                function_name = match.group(1)
                function_calls += 1
                function_names[function_name] += 1

                # Check if it's a finish action
                if function_name == "finish":
                    conversation_has_finish = True
                    # Automatically count finish actions as having thoughts
                    function_thoughts += 1

                # Check if it's a valid tool
                if function_name not in VALID_TOOLS:
                    invalid_tools.add(function_name)

                # Only check for thoughts if not a finish action
                if function_name != "finish":
                    thought_match = THOUGHT_PATTERN.search(content)
                    if thought_match and thought_match.group(1).strip():
                        function_thoughts += 1

            elif "<function=" in content:
                # Alternative pattern for function calls without </function> closing tag
                match = re.search(r"<function=([^>]+)>", content)
                if match:
                    function_name = match.group(1)
                    function_calls += 1
                    function_names[function_name] += 1

                    # Check if it's a finish action
                    if function_name == "finish":
                        conversation_has_finish = True

                    # Check if it's a valid tool
                    if function_name not in VALID_TOOLS:
                        invalid_tools.add(function_name)

        # Update has_finish_action if this conversation has a finish action
        if conversation_has_finish:
            finish_action_count += 1

    # Calculate function names sum to check if close to 1.0
    function_names_sum = sum(function_names.values()) / function_calls if function_calls > 0 else 0

    # Calculate thought percentage
    thought_percentage = (function_thoughts / function_calls * 100) if function_calls > 0 else 0

    # Check if all roles are valid
    valid_roles = all(
        role in ["human", "gpt", "function_call", "observation"] for role in roles.keys()
    )
    return {
        "dataset": dataset_name,
        "conversation_count": conversation_count,
        "roles": dict(roles),
        "function_calls": function_calls,
        "function_names": dict(function_names),
        "function_thoughts": function_thoughts,
        "function_names_sum": function_names_sum,
        "finish_action_count": finish_action_count,
        "len1_conversation_count": len1_conversation_count,
        "invalid_tools": list(invalid_tools),
        "thought_percentage": thought_percentage,
        "valid_roles": valid_roles,
    }


def create_roles_chart(results):
    """Create a stacked bar chart of roles per conversation."""
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
    df.plot(kind="bar", stacked=True, figsize=(12, 8))
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
    # Get all unique function names across datasets
    all_functions = set()
    for r in results:
        all_functions.update(r["function_names"].keys())

    # Create DataFrame for plotting
    data = []
    for r in results:
        row = {"dataset": r["dataset"]}
        # Use the actual function_calls count instead of just function_call role messages
        function_calls_count = r["function_calls"]
        for func in all_functions:
            row[func] = (
                r["function_names"].get(func, 0) / function_calls_count
                if function_calls_count > 0
                else 0
            )
        data.append(row)

    df = pd.DataFrame(data)
    df = df.set_index("dataset")

    # Save raw data to CSV
    df.to_csv("quality-control-results/function_names.csv")

    # Create stacked bar chart
    df.plot(kind="bar", stacked=True, figsize=(12, 8))
    plt.title("Function Names as Proportion of Total Function Calls")
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


def generate_markdown_table(results):
    """Generate separate markdown tables for passing and failing datasets."""
    # Initialize lists for passing and failing datasets
    passing_datasets = []
    failing_datasets = []

    # Process each result and categorize as passing or failing
    for result in results:
        dataset = result["dataset"]
        checks = {}

        # Check if function_names.csv adds up to close to 1.0
        function_names_sum = result["function_names_sum"]
        checks["function_names"] = 0.95 <= function_names_sum <= 1.05
        function_names_check = "✅" if checks["function_names"] else "❌"
        if function_names_sum == 0:
            function_names_check = "❌ (No functions)"
            checks["function_names"] = False

        # Check if finish action count > 80% or len1 conversation count > 80%
        finish_percent = (
            result["finish_action_count"] / result["conversation_count"] * 100
            if result["conversation_count"] > 0
            else 0
        )
        len1_percent = (
            result["len1_conversation_count"] / result["conversation_count"] * 100
            if result["conversation_count"] > 0
            else 0
        )

        if finish_percent > 80:
            has_finish = f"✅ (finish = {finish_percent:.1f}%)"
            checks["has_finish"] = True
        elif len1_percent > 80:
            has_finish = f"✅ (len1 = {len1_percent:.1f}%)"
            checks["has_finish"] = True
        else:
            has_finish = "❌"
            checks["has_finish"] = False

        # Check if only valid tools are used
        checks["valid_tools"] = not result["invalid_tools"] and result["function_calls"] > 0
        valid_tools = (
            "✅" if checks["valid_tools"] else f"❌ ({', '.join(result['invalid_tools'])})"
        )
        if result["function_calls"] == 0:
            valid_tools = "❌ (No functions)"
            checks["valid_tools"] = False

        # Check if >80% of functions have thoughts
        thought_percentage = result["thought_percentage"]
        checks["thought_check"] = thought_percentage >= 80 and result["function_calls"] > 0
        thought_check = "✅" if checks["thought_check"] else f"❌ ({thought_percentage:.1f}%)"
        if result["function_calls"] == 0:
            thought_check = "❌ (No functions)"
            checks["thought_check"] = False

        # Check if roles are valid
        checks["valid_roles"] = result["valid_roles"]
        valid_roles = "✅" if checks["valid_roles"] else "❌"

        # Create dataset entry
        dataset_entry = {
            "dataset": dataset,
            "function_names_check": function_names_check,
            "has_finish": has_finish,
            "valid_tools": valid_tools,
            "thought_check": thought_check,
            "valid_roles": valid_roles,
            "all_pass": all(checks.values()),
        }

        # Add to appropriate list
        if dataset_entry["all_pass"]:
            passing_datasets.append(dataset_entry)
        else:
            failing_datasets.append(dataset_entry)

    # Create markdown content
    markdown = "# SFT Quality Control Results\n\n"

    # Table for passing datasets
    markdown += "## Passing Datasets\n\n"
    if passing_datasets:
        markdown += "| Dataset | Function Names Sum to 1.0 | Finish or Len1 | Only Valid Tools | >80% Functions Have Thoughts | Valid Roles |\n"
        markdown += "|---------|-------------------------|-------------------|-----------------|----------------------------|------------|\n"

        for entry in passing_datasets:
            markdown += f"| {entry['dataset']} | {entry['function_names_check']} | {entry['has_finish']} | {entry['valid_tools']} | {entry['thought_check']} | {entry['valid_roles']} |\n"
    else:
        markdown += "No datasets passed all quality checks.\n\n"

    # Table for failing datasets
    markdown += "\n## Failing Datasets\n\n"
    if failing_datasets:
        markdown += "| Dataset | Function Names Sum to 1.0 | Finish or Len1 | Only Valid Tools | >80% Functions Have Thoughts | Valid Roles |\n"
        markdown += "|---------|-------------------------|-------------------|-----------------|----------------------------|------------|\n"

        for entry in failing_datasets:
            markdown += f"| {entry['dataset']} | {entry['function_names_check']} | {entry['has_finish']} | {entry['valid_tools']} | {entry['thought_check']} | {entry['valid_roles']} |\n"
    else:
        markdown += "All datasets passed all quality checks.\n\n"

    # Write to file
    with open("quality-control-results/sft_quality_check.md", "w", encoding="utf-8") as f:
        f.write(markdown)

    return markdown


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
        print(f"  Function names sum: {result['function_names_sum']:.2f}")
        print(
            f"  Finish action count: {result['finish_action_count']}/{result['conversation_count'] * 100:.1f}%"
        )
        print(
            f"  Len1 conversation count: {result['len1_conversation_count']}/{result['conversation_count'] * 100:.1f}%"
        )
        print(f"  Invalid tools: {result['invalid_tools']}")
        print(f"  Thought percentage: {result['thought_percentage']:.1f}%")
        print(f"  Valid roles: {result['valid_roles']}")

    # Create visualizations
    create_roles_chart(results)
    create_function_names_chart(results)
    create_function_thought_chart(results)

    # Generate markdown table
    markdown_table = generate_markdown_table(results)
    print("\nMarkdown table generated:")
    print(markdown_table)

    print("\nAnalysis complete. Generated files in quality-control-results/:")
    print("- roles_per_conversation.png and roles_per_conversation.csv")
    print("- function_names.png and function_names.csv")
    print("- function_thought_percentage.png and function_thought_percentage.csv")
    print("- sft_quality_check.md")


if __name__ == "__main__":
    main()
