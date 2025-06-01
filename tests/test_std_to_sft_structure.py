def test_main_function_exists():
    """Test that the main function exists in the std_to_sft module."""
    # Read the file content directly
    with open("scripts/std_to_sft.py", "r") as f:
        content = f.read()

    # Check if main function exists
    assert "def main():" in content, "main function should exist in std_to_sft.py"


def test_if_main_block_exists():
    """Test that the if __name__ == '__main__' block exists in the std_to_sft module."""
    # Read the file content
    with open("scripts/std_to_sft.py", "r") as f:
        content = f.read()

    # Check if the if __name__ == '__main__' block exists
    assert 'if __name__ == "__main__":' in content, (
        "if __name__ == '__main__' block should exist in std_to_sft.py"
    )

    # Check if main() is called in the if block
    assert 'if __name__ == "__main__":\n    main()' in content, (
        "main() should be called in the if __name__ == '__main__' block"
    )
