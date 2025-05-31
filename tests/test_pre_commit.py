import os
import subprocess


def test_pre_commit_config_exists():
    """Test that the pre-commit configuration file exists."""
    assert os.path.exists(".pre-commit-config.yaml"), ".pre-commit-config.yaml should exist"


def test_ruff_config_exists():
    """Test that the ruff configuration exists in pyproject.toml."""
    assert os.path.exists("pyproject.toml"), "pyproject.toml should exist"


def test_mypy_config_exists():
    """Test that the mypy configuration file exists."""
    assert os.path.exists("mypy.ini"), "mypy.ini should exist"


def test_pre_commit_hooks_pass():
    """Test that pre-commit hooks pass on a sample file."""
    # Create a temporary Python file
    with open("test_file.py", "w") as f:
        f.write('def test_function():\n    """Test function docstring."""\n    return True\n')

    try:
        # Run pre-commit on the file
        result = subprocess.run(
            ["pre-commit", "run", "--files", "test_file.py"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"pre-commit failed: {result.stderr}"
    finally:
        # Clean up the temporary file
        if os.path.exists("test_file.py"):
            os.remove("test_file.py")
