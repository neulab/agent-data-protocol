#!/bin/bash

# Print a message to indicate the script is running
echo "Setting up pre-commit hooks for agent-data-collection..."

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "pip is not installed. Installing pip..."
    sudo apt-get update
    sudo apt-get install -y python3-pip
fi

# Install pre-commit if not already installed
if ! command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit..."
    pip install pre-commit
else
    echo "pre-commit is already installed."
fi

# Install the pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Update the pre-commit hooks to the latest versions
echo "Updating pre-commit hooks..."
pre-commit autoupdate

echo "Setup complete! pre-commit hooks are now installed and ready to use."
