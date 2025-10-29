#!/bin/bash
# Script to install pre-commit hooks

set -e

# Install pre-commit if not already installed
if ! command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit..."
    pip install pre-commit
fi

# Install the pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Install the pre-commit hooks for commit-msg
pre-commit install --hook-type commit-msg

echo "Pre-commit hooks installed successfully!"
