import re


def has_thought(content):
    """Check if the content has any text besides function calls."""
    # Check for function call pattern
    has_function_call = bool(re.search(r"ACTION:\s*\n```[a-zA-Z0-9_]+", content))

    # Remove the function call pattern from the content
    cleaned_content = re.sub(r"ACTION:\s*\n```[a-zA-Z0-9_]+.*?```", "", content, flags=re.DOTALL)

    # Also check for THOUGHT: pattern
    has_thought_marker = bool(re.search(r"THOUGHT:", content, re.IGNORECASE))

    # Remove whitespace and check if there's any content left
    cleaned_content = cleaned_content.strip()

    # If there's text content besides function calls, or no function call at all, or explicit THOUGHT: marker
    return bool(cleaned_content) or not has_function_call or has_thought_marker
