def has_thought(content):
    """Check if the content has any text besides function calls."""
    # Simply check if there's any text content besides the function call patterns
    # This is a simplified approach as requested in the review
    
    # Remove common function call patterns
    cleaned_content = content
    
    # Remove code blocks
    cleaned_content = re.sub(r"```[^`]+```", "", cleaned_content)
    
    # Remove execute tags
    cleaned_content = re.sub(r"<execute_\w+>.*?</execute_\w+>", "", cleaned_content, flags=re.DOTALL)
    
    # Remove antml function calls
    cleaned_content = re.sub(r"<function_calls>.*?